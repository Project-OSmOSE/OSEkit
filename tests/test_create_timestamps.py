from osmose_package.OScreate_timestamps import *
import re

def test_convert_template_to_re():
    raw_all = "".join(converter.keys())
    simple_template = "%Y/%m/%d"
    simple_text = "sample_file_2017/02/24.txt"
    invalid_simple_text = "sample_file_2049/25/01"
    complex_template = "y_%Y-m_%m, %I%p."
    complex_text = " y_2017-m_02, 11AM%"

    assert convert_template_to_re(raw_all) == "".join(converter.values())
    simple_res = convert_template_to_re(simple_template)
    assert re.search(simple_res, simple_text)[0] == "2017/02/24"
    assert re.search(simple_res, invalid_simple_text) == None
    complex_res = convert_template_to_re(complex_template)
    assert re.serach(complex_res, complex_text)[0] == "y_2017-m_02, 11AM"

# a monkeypatch
def test_write_timestamp():
    pass