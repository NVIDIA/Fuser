from __future__ import annotations
__all__: list = ['NvfuserVersion', 'Version']
class NvfuserVersion(str):
    @staticmethod
    def __eq__(x, y, method = '__eq__'):
        ...
    @staticmethod
    def __ge__(x, y, method = '__ge__'):
        ...
    @staticmethod
    def __gt__(x, y, method = '__gt__'):
        ...
    @staticmethod
    def __le__(x, y, method = '__le__'):
        ...
    @staticmethod
    def __lt__(x, y, method = '__lt__'):
        ...
    @classmethod
    def _convert_to_version(cls, ver: typing.Any) -> <nvfuser.nvfuser_version._LazyImport object>:
        ...
    def _cmp_version(self, other: typing.Any, method: str) -> <nvfuser.nvfuser_version._LazyImport object>:
        ...
class _LazyImport:
    """
    Wraps around classes lazy imported from packaging.version
        Output of the function v in following snippets are identical:
           from packaging.version import Version
           def v():
               return Version('1.2.3')
        and
           Version = _LazyImport('Version')
           def v():
               return Version('1.2.3')
        The difference here is that in later example imports
        do not happen until v is called
        
    """
    def __call__(self, *args, **kwargs):
        ...
    def __init__(self, cls_name: str) -> None:
        ...
    def __instancecheck__(self, obj):
        ...
    def get_cls(self):
        ...
Version: _LazyImport  # value = <nvfuser.nvfuser_version._LazyImport object>
__version__: NvfuserVersion  # value = '0.1.5+git6830ed2'
_version_str: str = '0.1.5+git6830ed2'
cmp_method: str = '__le__'
