import ssms
from ssms.config import model_config

class TestModelConfig:
    def test_model_config_dict_type(self):
        assert isinstance(model_config, ssms.config.CopyOnAccessDict)


    def test_model_config_copy_on_access(self):
        list_params = model_config["shrink_spot_simple_extended"]["params"]
        list_params.append("p_outlier")
        assert "p_outlier" not in model_config["shrink_spot_simple_extended"]["params"]
