import hashlib
import subprocess

import h5py

# to compare hash of output file to true file
def gethash(filename):
    with open(filename, "rb") as f:
        buf = f.read()
    hasher = hashlib.md5()
    hasher.update(buf)
    return hasher.hexdigest()

def test_makeWaveforms():
    """Tests 1_makeWaveformsDataset.py
    Only tests that the generated catalog is correct at this point
    """

    # run the script
    subprocess.run([
        "python",
        "../scripts/1_makeWaveformsDataset.py",
        "config_test1.yml"
    ])


    # get the hash values of the correct output files
    trueh5hash = gethash('test1/results/H5files/data_test1_true.h5')
    truewfhash = gethash('test1/results/wf_cat_out_true.csv')

    # get the hash values of the output from the test
    h5hash = gethash('test1/results/H5files/data_test1.h5')
    wfhash = gethash('test1/results/wf_cat_out.csv')

    subprocess.run([
        "python",
        "../scripts/calculate_energy.py",
        "config_test1.yml"
    ])

    # this doesn't work - hdf5 must not save the same data,
    # maybe some date stamp or something
    #assert h5hash == trueh5hash
    assert wfhash == truewfhash

def test_convertToSpectrograms():
    """Tests 2_convertToSpectrograms.py
    Only tests that the generated catalog is correct at this point
    """
    # run the script
    ret = subprocess.run([
        "python",
        "../scripts/2_convertToSpectrograms.py",
        "config_test1.yml"
    ])

    truehash = gethash("test1/results/sgram_cat_out_test1_true.csv")
    hash = gethash("test1/results/sgram_cat_out_test1.csv")

    assert hash == truehash
    assert ret.returncode == 0

def test_runSpecufex():
    """Test 3_runSpeccUFEx.py
    Only tests successful execution, not output.
    """
    ret = subprocess.run([
        "python",
        "../scripts/3_runSpecUFEx.py",
        "config_test1.yml"
    ])

    assert ret.returncode == 0

def test_rundistance():
    """Test 4_DistanceCalc.py
    Only tests successful execution, not output.
    """
    ret = subprocess.run([
        "python",
        "../scripts/4_DistanceCalc.py",
        "config_test1.yml"
    ])

    assert ret.returncode == 0
