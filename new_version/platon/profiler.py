# This is a dummy file to make the @profile decorator work with kernprof.
# It allows us to add the decorator to the code for profiling without causing
# an import error when running the code normally.

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

if 'profile' not in builtins.__dict__:
    builtins.__dict__['profile'] = lambda f: f