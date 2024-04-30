from OSmOSE.scales.custom_scale import CustomScale
from OSmOSE.scales.scale_serializer import ScaleSerializer
import numpy as np
import pytest

configs_to_test = ["porp_delph","Dual_LF_HF","Audible"]
important_freqs = [22050,44100,156250,312500]

def test_scale_serializer():
    serializer = ScaleSerializer()

    assert isinstance(serializer.get_scale("porp_delph",sr=312500), CustomScale), "Error in porp_delph scale"
    assert isinstance(serializer.get_scale("Audible",sr=312500), CustomScale), "Error in Audible scale"
    assert isinstance(serializer.get_scale("Dual_LF_HF",sr=312500), CustomScale), "Error in Dual_LF_HF scale"

    try:
        serializer.get_scale("non_existent_scale",sr=312500)
    except ValueError:
        pass
    else:
        print("Error: non_existent_scale should raise a ValueError")

def test_custom_scales():
    for config in configs_to_test:
        print(config)
        scale = ScaleSerializer().get_scale(config,sr=312500)
        assert isinstance(scale, CustomScale)
        assert np.isclose(scale.map_freq2scale(0),0,rtol=1e-12)
        # Test frequency to scale mapping
        assert np.isclose(scale.map_freq2scale(scale.frequencies[0]),scale.coefficients[0],rtol=1e-12)
        # Test scale to frequency mapping
        assert np.isclose(scale.map_scale2freq(0) , 0,rtol=1e-12)
        assert np.isclose(scale.map_scale2freq(scale.coefficients[0]) , scale.frequencies[0],rtol=1e-12)
            
        test_box = [0,15,2,3]
        test_box2 = [0.001,0.101,0,156250]

        a,b,c,d = scale.bbox2scale(test_box[0],test_box[1],test_box[2],test_box[3])
        e,f,g,h = scale.bbox2scale(test_box2[0],test_box2[1],test_box2[2],test_box2[3])


        assert np.isclose(scale.scale2bbox(a,b,c,d)[0],test_box[0],rtol=1e-12)
        assert np.isclose(scale.scale2bbox(a,b,c,d)[1],test_box[1],rtol=1e-12)
        assert np.isclose(scale.scale2bbox(a,b,c,d)[2],test_box[2],rtol=1e-12)
        assert np.isclose(scale.scale2bbox(a,b,c,d)[3],test_box[3],rtol=1e-12)
        assert np.isclose(scale.scale2bbox(e,f,g,h)[0],test_box2[0],rtol=1e-12)
        assert np.isclose(scale.scale2bbox(e,f,g,h)[1],test_box2[1],rtol=1e-12)
        assert np.isclose(scale.scale2bbox(e,f,g,h)[2],test_box2[2],rtol=1e-12)
        if config != "Audible":
            assert np.isclose(scale.scale2bbox(e,f,g,h)[3],test_box2[3],rtol=1e-12)
        else:
            assert np.isclose(scale.scale2bbox(e,f,g,h)[3],scale.frequencies[0],rtol=1e-12)