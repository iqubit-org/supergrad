"""
The code is from `jax._src.numpy.util `.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import re
import textwrap
from typing import Any, NamedTuple, TypeVar

from jax._src import config

_T = TypeVar("_T")
_parameter_break = re.compile("\n(?=[A-Za-z_])")
_section_break = re.compile(r"\n(?=[^\n]{3,15}\n-{3,15})", re.MULTILINE)
_numpy_signature_re = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\([\w\W]*?\)$',
                                 re.MULTILINE)
_versionadded = re.compile(r'^\s+\.\.\s+versionadded::', re.MULTILINE)
_docreference = re.compile(r':doc:`(.*?)\s*<.*?>`')


class ParsedDoc(NamedTuple):
    """
  docstr: full docstring
  signature: signature from docstring.
  summary: summary from docstring.
  front_matter: front matter before sections.
  sections: dictionary of section titles to section content.
  """
    docstr: str | None
    signature: str = ""
    summary: str = ""
    front_matter: str = ""
    sections: dict[str, str] = {}


def _parse_numpydoc(docstr: str | None) -> ParsedDoc:
    """Parse a standard numpy-style docstring.
  Args:
    docstr: the raw docstring from a function
  Returns:
    ParsedDoc: parsed version of the docstring
  """
    if docstr is None or not docstr.strip():
        return ParsedDoc(docstr)
    # Remove any :doc: directives in the docstring to avoid sphinx errors
    docstr = _docreference.sub(lambda match: f"{match.groups()[0]}", docstr)
    signature, body = "", docstr
    match = _numpy_signature_re.match(body)
    if match:
        signature = match.group()
        body = docstr[match.end():]
    firstline, _, body = body.partition('\n')
    body = textwrap.dedent(body.lstrip('\n'))
    match = _numpy_signature_re.match(body)
    if match:
        signature = match.group()
        body = body[match.end():]
    summary = firstline
    if not summary:
        summary, _, body = body.lstrip('\n').partition('\n')
        body = textwrap.dedent(body.lstrip('\n'))
    front_matter = ""
    body = "\n" + body
    section_list = _section_break.split(body)
    if not _section_break.match(section_list[0]):
        front_matter, *section_list = section_list
    sections = {section.split('\n', 1)[0]: section for section in section_list}
    return ParsedDoc(docstr=docstr,
                     signature=signature,
                     summary=summary,
                     front_matter=front_matter,
                     sections=sections)


def _parse_parameters(body: str) -> dict[str, str]:
    """Parse the Parameters section of a docstring."""
    title, underline, content = body.split('\n', 2)
    assert title == 'Parameters'
    assert underline and not underline.strip('-')
    parameters = _parameter_break.split(content)
    return {p.partition(' : ')[0].partition(', ')[0]: p for p in parameters}


def implements(
    original_fun: Callable[..., Any] | None,
    update_doc: bool = True,
    sections: Sequence[str] = ('Parameters', 'Returns', 'References'),
    module: str | None = None,
) -> Callable[[_T], _T]:
    """Decorator for SuperGrad functions which implement a specified NumPy function.
  This mainly contains logic to copy and modify the docstring of the original
  function. In particular, if `update_doc` is True, parameters listed in the
  original function that are not supported by the decorated function will
  be removed from the docstring. For this reason, it is important that parameter
  names match those in the original numpy function.
  Args:
    original_fun: The original function being implemented
    update_doc: whether to transform the numpy docstring to remove references of
      parameters that are supported by the numpy version but not the JAX version.
      If False, include the numpy docstring verbatim.
    sections: a list of sections to include in the docstring. The default is
      ["Parameters", "Returns", "References"]
    module: an optional string specifying the module from which the original function
      is imported. This is useful for objects such as ufuncs, where the module cannot
      be determined from the original function itself.
  """

    def decorator(wrapped_fun):
        wrapped_fun.__np_wrapped__ = original_fun
        # Allows this pattern: @implements(getattr(np, 'new_function', None))
        if original_fun is None:
            return wrapped_fun
        docstr = getattr(original_fun, "__doc__", None)
        name = getattr(original_fun, "__name__",
                       getattr(wrapped_fun, "__name__", str(wrapped_fun)))
        try:
            mod = module or original_fun.__module__
        except AttributeError:
            if config.enable_checks.value:
                raise ValueError(
                    f"function {original_fun} defines no __module__; pass module keyword to implements()."
                )
        else:
            name = f"{mod}.{name}"
        if docstr:
            try:
                parsed = _parse_numpydoc(docstr)
                if update_doc and 'Parameters' in parsed.sections:
                    code = getattr(
                        getattr(wrapped_fun, "__wrapped__", wrapped_fun),
                        "__code__", None)
                    # Remove unrecognized parameter descriptions.
                    parameters = _parse_parameters(
                        parsed.sections['Parameters'])
                    parameters = {
                        p: desc
                        for p, desc in parameters.items()
                        if (code is None or p in code.co_varnames)
                    }
                    if parameters:
                        parsed.sections['Parameters'] = (
                            "Parameters\n"
                            "----------\n" + "\n".join(
                                _versionadded.split(desc)[0].rstrip()
                                for p, desc in parameters.items()))
                    else:
                        del parsed.sections['Parameters']
                docstr = parsed.summary.strip() + "\n" if parsed.summary else ""
                docstr += f"\nSuperGrad-supported implementation of :func:`{name}`.\n"
                docstr += "\n*Original docstring below.*\n"
                # We remove signatures from the docstrings, because they redundant at best and
                # misleading at worst: e.g. JAX wrappers don't implement all ufunc keyword arguments.
                # if parsed.signature:
                #   docstr += "\n" + parsed.signature.strip() + "\n"
                if parsed.front_matter:
                    docstr += "\n" + parsed.front_matter.strip() + "\n"
                kept_sections = (
                    content.strip()
                    for section, content in parsed.sections.items()
                    if section in sections)
                if kept_sections:
                    docstr += "\n" + "\n\n".join(kept_sections) + "\n"
            except:
                if config.enable_checks.value:
                    raise
                docstr = original_fun.__doc__
        wrapped_fun.__doc__ = docstr
        for attr in ['__name__', '__qualname__']:
            try:
                value = getattr(original_fun, attr)
            except AttributeError:
                pass
            else:
                setattr(wrapped_fun, attr, value)
        return wrapped_fun

    return decorator
