is_simple_core =False
if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import no_grad
    from dezero.core_simple import using_config
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    import dezero.cuda 
    from dezero.models import Model
    from dezero.layers import Layer
    import dezero.datasets
    from dezero.core import Config
    from dezero.core import test_mode
    import dezero.functions_conv

setup_variable()