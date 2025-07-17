import numpy as np
from pandas import DataFrame

from ..utils.strings import clean_string, remove_newline
character_list = ["[", "]", "\n", "#"]

from ..visualization.indexation import indexation_plot

# ======================== Functions to parse the file entries ========================

def read_number_indexed_spots(fitfile_obj, file, line):
    """Read the number of indexed spots from file and add attribute to the object FitFile
    Adds attribute number_indexed_spots to the class.
    """
    # Sample entry:
    # "#Number of indexed spots: 185"
    fitfile_obj.number_indexed_spots = int(line.split()[-1])

def read_element(fitfile_obj, file, line):
    "Read the element from file and add attribute to the object FitFile"
    # Sample entry:
    # "#Element"
    # "#4H-SiC"
    line = clean_string(file.readline(), character_list)
    fitfile_obj.element = line
    
def read_grain_index(fitfile_obj, file, line):
    "Read the grain index from file and add attribute to the object FitFile"
    line = clean_string(file.readline(), character_list)
    fitfile_obj.grain_index = line

def read_mean_pixel_deviation(fitfile_obj, file, line):
    "Read the mean pixel deviation from file and add attribute to the object FitFile"
    # Sample entry:
    # "#Mean Deviation(pixel): 0.174"
    fitfile_obj.mean_pixel_deviation = float(line.split()[-1])
    
def read_peaks_data(fitfile_obj, file, line):
    "Read the peak list with its properties from file and add attribute to the object FitFile"
    # sample entry:
    # "##spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev"
    # "0.000000 61195.040000 0.000000 0.000000 6.000000 85.780427 0.861945 1023.590000 1196.750000 5.431640 0.000000 0.086716"
    peaklist = np.loadtxt(fitfile_obj.filename)[:, 1:] #skip first column with the spot index, as it
                                                       # will be included automatically within the dataframe
    nb_peaks = len(peaklist)
    file_columns = line.split()[1:] # removing spot_index
    dataframe_columns = [dataframe_name[col] for col in file_columns]
    fitfile_obj.number_indexed_spots = nb_peaks
    # convert_dtypes infers most suitable dtype. This will make h, k, l and grain idx integers
    fitfile_obj.peaklist = DataFrame(data=peaklist, columns=dataframe_columns).convert_dtypes()
    
    # skip lines containing the peaks
    for _ in range(nb_peaks):
        next(file)
    

def read_UB(fitfile_obj, file, line):
    "Read the UB matrix from file and add attribute to the object FitFile"
    # sample entry:
    #
    # "#UB matrix in q= (UB) B0 G*"
    # "#[[-0.639212335 -0.35898807  -0.68001994 ]"
    # "# [ 0.480559684 -0.876755415  0.011069849]"
    # "# [-0.600391523 -0.319949054  0.73197225 ]]"
    
    UB  = []
    
    for line_number in range(3):
        # line is of type list[str]
        line = clean_string(file.readline(), character_list).split()
        # append the type-corrected list of elements
        UB.append([float(elem) for elem in line])
    
    fitfile_obj.UB = np.array(UB)
    
def read_B0(fitfile_obj, file, line):
    "Read the B0 matrix from file and add attribute to the object FitFile"
    # sample entry:
    #
    # "#B0 matrix in q= UB (B0) G*"
    # "#[[ 0.37575676  0.18787838 -0.        ]"
    # "# [ 0.          0.3254149  -0.        ]"
    # "# [ 0.          0.          0.09947279]]"
    
    B0 = []
    
    for line_number in range(3):
        # line is of type list[str]
        line = clean_string(file.readline(), character_list).split()
        # append the type-corrected list of elements
        B0.append([float(elem) for elem in line])
    
    fitfile_obj.B0 = np.array(B0)

def read_UBB0(fitfile_obj, file, line):
    "Read the UBB0 matrix from file and add attribute to the object FitFile"
    # sample entry:
    # 
    # "#UBB0 matrix in q= (UB B0) G* i.e. recip. basis vectors are columns in LT frame: astar = UBB0[:,0], bstar = UBB0[:,1], cstar = UBB0[:,2]. (abcstar as columns on xyzlab1, xlab1 = ui, ui = unit vector along incident beam) "
    # "#[[-0.24018836 -0.23691425 -0.06764348]"
    # "#[ 0.18057355 -0.1950225   0.00110115]"
    # "#[-0.22560118 -0.21691678  0.07281133]]"
    
    UBB0  = []
    
    for line_number in range(3):
        # line is of type list[str]
        line = clean_string(file.readline(), character_list).split()
        # append the type-corrected list of elements
        UBB0.append([float(elem) for elem in line])
    
    fitfile_obj.UBB0 = np.array(UBB0)

def read_euler_angles(fitfile_obj, file, line):
    "Read the euler angles from file and add attribute to the object FitFile"
    # sample entry:
    #
    # "#Euler angles phi theta psi (deg)"
    # "#[298.044  42.864  89.072]"
    
    line = clean_string(file.readline(), character_list).split()
    
    fitfile_obj.euler_angles = np.array([float(element) for element in line])

def read_deviatoric_strain_crystal_frame(fitfile_obj, file, line):
    "Read the deviatoric strain tensor in the crystal frame from file and add attribute to the object FitFile"
    # sample entry:
    # 
    # "#deviatoric strain in direct crystal frame (10-3 unit)"
    # "#[[-0.18 -0.07 -0.17]"
    # "# [-0.07 -0.36 -0.23]"
    # "# [-0.17 -0.23  0.54]]"
    
    dev_crystal  = []
    
    for line_number in range(3):
        # line is of type list[str]
        line = clean_string(file.readline(), character_list).split()
        # append the type-corrected list of elements
        dev_crystal.append([float(elem) * 1e-3 for elem in line])
    
    fitfile_obj.deviatoric_strain_crystal_frame = np.array(dev_crystal)

def read_deviatoric_strain_sample_frame(fitfile_obj, file, line):
    "Read the deviatoric strain tensor in the sample frame from file and add attribute to the object FitFile"
    # sample entry:
    # 
    # "#deviatoric strain in sample2 frame (10-3 unit)"
    # "#[[-0.32 -0.11  0.21]"
    # "# [-0.11 -0.24  0.17]"
    # "# [ 0.21  0.17  0.56]]"
    
    dev_sample  = []
    
    for line_number in range(3):
        # line is of type list[str]
        line = clean_string(file.readline(), character_list).split()
        # append the type-corrected list of elements
        dev_sample.append([float(elem) * 1e-3 for elem in line])
    
    fitfile_obj.deviatoric_strain_sample_frame = np.array(dev_sample)
    
def read_new_lattice_parameters(fitfile_obj, file, line):
    "Read the new lattice parameters from file and add attribute to the object FitFile"
    # sample entry:
    # "#new lattice parameters"
    # "#[  3.073       3.0727618  10.0603084  90.0125899  90.0198677 120.0106641]"
    line = clean_string(file.readline(), character_list).split()
    
    fitfile_obj.new_lattice_parameters = np.array([float(element) for element in line])
    
def read_camera_dict(fitfile_obj, file, line):
    # sample entry:
    #
    # "#CCDLabel"
    # "#sCMOS"
    # "#DetectorParameters"
    # "#[77.98, 1039.97, 1126.52, 0.433, 0.33]"
    # "#pixelsize"
    # "#0.0734"
    # "#Frame dimensions"
    # "#[2018.0, 2016.0]"
    
    camera_dict = {}
    
    line = clean_string(file.readline(), character_list)
    camera_dict["CCDLabel"] = line
    
    file.readline() # catch "DetectorParameters" string
    line = clean_string(file.readline(), character_list).split(", ")
    camera_dict["DetectorParameters"] = np.array([float(element) for element in line])
    
    file.readline() # catch "pixelsize" string
    line = clean_string(file.readline(), character_list)
    camera_dict["pixelsize"] = float(line)
    
    file.readline() # catch "Frame dimensions" string
    line = clean_string(file.readline(), character_list).split(", ")
    camera_dict["framedim"] = (line[0], line[1])
    
    fitfile_obj.CCDdict = camera_dict

# ============================== File entries dictionary ==============================
#
# key: file entry to read
# value: function to read that entry
file_entries = {
    "Number of indexed spots": read_number_indexed_spots,
    "Element": read_element,
    "grainIndex": read_grain_index,
    "Mean Deviation(pixel)": read_mean_pixel_deviation,
    "spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev": read_peaks_data,
    "spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz": read_peaks_data,
    "spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz grainindex": read_peaks_data,
    "UB matrix in q= (UB) B0 G* ": read_UB,
    "B0 matrix in q= UB (B0) G*": read_B0,
    "UBB0 matrix in q= (UB B0) G* i.e. recip. basis vectors are columns in LT frame: astar = UBB0[:,0], bstar = UBB0[:,1], cstar = UBB0[:,2]. (abcstar as columns on xyzlab1, xlab1 = ui, ui = unit vector along incident beam)": read_UBB0,
    "Euler angles phi theta psi (deg)": read_euler_angles,
    "deviatoric strain in direct crystal frame (10-3 unit)": read_deviatoric_strain_crystal_frame,
    "deviatoric strain in sample2 frame (10-3 unit)": read_deviatoric_strain_sample_frame,
    "new lattice parameters": read_new_lattice_parameters,
    "CCDLabel": read_camera_dict,
}

# ========== Name mapping from file to dataframe columns in FitFile.peaklist ==========

dataframe_name = {
    "intensity": "Intensity",
    "Intensity": "Intensity",
    "h": "h",
    "k": "k",
    "l": "l",
    "pixDev": "pixel dev",
    "PixDev": "pixel dev",
    "Energy": "Energy",
    "energy(keV)": "Energy",
    "Xexp": "Xexp",
    "Yexp": "Yexp",
    "2theta": "2θexp",
    "Chi": "χexp",
    "2theta_exp": "2θexp",
    "chi_exp": "χexp",
    "Xtheo": "Xtheo",
    "Ytheo": "Ytheo",
    "2theta_theo": "2θtheo",
    "chi_theo": "χtheo",
    "Qx": "Qx",
    "Qy": "Qy",
    "Qz": "Qz",
    "grainindex": "grain idx",
    "GrainIndex": "grain idx"
}
    
# ========================= Functions to parse the entire file ========================
    
def read_file_header(fitfile_obj, file):
    # Sample header
    # ------------------------------------------------------------------------------------
    #Strain and Orientation Refinement from experimental file: /gpfs/jazzy/data/bm32/inhouse/STAFF/SERGIOB/20240702 - ihma513 (SiC)/sample1_tip/corfiles/img_0000.cor
    #File created at Mon Sep  2 14:51:41 2024 with indexingSpotsSet.py
    #Number of indexed spots: 185
    #Element
    #4H-SiC
    #grainIndex
    #G_0
    #Mean Deviation(pixel): 0.174
    
    line = remove_newline(file.readline())
    fitfile_obj.corfile = line.split(": ")[-1]
    
    line = remove_newline(file.readline())
    fitfile_obj.timestamp, fitfile_obj.software = line.lstrip("#File created at ").split(" with ")

def read_file_body(fitfile_obj, file):
    """Compare each read line to the dictionary and call the corresponding function to parse it"""
    # Dictionary whose keys are the file entries, and the values are the functions to read them
    
    line = file.readline()
    line = clean_string(line, ["\n", "#"])
    
    while line != "\n" and line != "":
        
        try:
            file_entries[line](fitfile_obj, file, line)
            
            line = file.readline()
            line = clean_string(line, ["\n", "#"])
            
        except KeyError:
            # for entries that have both text and data in the same line
            try:
                line_beginning = line.split(":")[0]
                file_entries[line_beginning](fitfile_obj, file, line)
                
                line = file.readline()
                line = clean_string(line, ["\n", "#"])
            
            # if this doesn't work either, rip    
            except KeyError:
                print(f"Could not read line: \n{line}")
                
                # moving on
                line = file.readline()
                line = clean_string(line, ["\n", "#"])

class FitFile:
    """Object containing the properties of a fit file

    Attributes
    ----------
    filename : str
        path to the parsed .fit file
    corfile  : str
        path to the .cor file containing the data used for indexing 
    UB: np.ndarray
        Orientation matrix of the crystal. Shape (3,3)
    B0: np.ndarray
        fill_docstring
    UBB0: np.ndarray
        fill_docstring
    element: str
        Name of the material whose lattice parameters are used for indexing
    euler_angles: np.ndarray
        fill_docstring
   
    GrainIndex: str
        fill_docstring
    mean_pixel_deviation: float
        fill_docstring
    number_indexed_spots: int
        fill_docstring
    
    new_lattice_parameters: np.ndarray
        fill_docstring
        
    a_prime: np.ndarray
        fill_docstring
    b_prime: np.ndarray
        fill_docstring
    c_prime: np.ndarray
        fill_docstring
    astar_prime: np.ndarray
        fill_docstring
    bstar_prime: np.ndarray
        fill_docstring
    cstar_prime: np.ndarray
        fill_docstring
    boa: np.float64
        Lattice parameter ratio b/a
    coa: np.float64
        Lattice parameter ratio c/a

    deviatoric_strain_sample_frame: np.ndarray
        fill_docstring
    deviatoric_strain_crystal_frame: np.ndarray
        fill_docstring
    
    peak: dict
        keys   -> miller indices (e.g. '0 0 6', or '-1 0 11')
        values -> spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev
    
    software: str
        fill_docstring
    timestamp: str
        fill_docstring
    CCDdict: dict
        fill_docstring
    Functions:
    """
    
    def __init__(self, filename: str, verbose: bool = False):        
        with open(filename, "r") as file:
            self.filename = filename
            
            read_file_header(self, file)
            read_file_body(self, file)
            self._compute_reciprocal_space()
            
    def _compute_reciprocal_space(self):
        # some extra calculations to get the direct and reciprocal lattice basis vector
        # NOTE: the scale of the lattice basis vector is UNKNOWN !!!
        #       they are given here with a arbitrary scale factor
        if not hasattr(self, "UBB0"):
            self.UBB0 = np.dot(self.UB, self.B0)
            
        try:
            self.astar_prime = self.UBB0[:, 0]
            self.bstar_prime = self.UBB0[:, 1]
            self.cstar_prime = self.UBB0[:, 2]

            self.a_prime = np.cross(self.bstar_prime, self.cstar_prime) / np.dot(self.astar_prime, np.cross(self.bstar_prime, self.cstar_prime))
            self.b_prime = np.cross(self.cstar_prime, self.astar_prime) / np.dot(self.bstar_prime, np.cross(self.cstar_prime, self.astar_prime))
            self.c_prime = np.cross(self.astar_prime, self.bstar_prime) / np.dot(self.cstar_prime, np.cross(self.astar_prime, self.bstar_prime))

            self.boa = np.linalg.linalg.norm(self.b_prime) / np.linalg.linalg.norm(self.a_prime)
            self.coa = np.linalg.linalg.norm(self.c_prime) / np.linalg.linalg.norm(self.a_prime)
        except ValueError:
            print("could not compute the reciprocal space from the UBB0")

    @property
    def peak_positions(self):
        return self.peaklist[["Xexp", "Yexp"]]
    
    @property
    def peak_info(self):
        return self.peaklist[["h", "k", "l", "Xexp", "Yexp", "2θexp", "χexp", "Intensity"]]
    
    # Methods that will be useful for FitFileSeries
    @property
    def lattice_parameters(self):
        columns = ["a'", "b'", "c'", "α'", "β'", "γ'", "b/a", "c/a"]
        data = np.concatenate(
                (self.new_lattice_parameters, self.boa, self.coa)
            ).reshape(1,-1)
        
        return DataFrame(data=data, columns=columns)
    
    @property
    def direct_lattice(self):
        columns = [
            "a'x", "a'y", "a'z",
            "b'x", "b'y", "b'z",
            "c'x", "c'y", "c'z"
        ]
        data = np.concatenate(
            (self.a_prime, self.b_prime, self.c_prime)
            ).reshape(1,-1)
        
        return DataFrame(data=data, columns=columns)
    
    @property
    def reciprocal_lattice(self):
        columns = [
            "a*'x", "a*'y", "a*'z",
            "b*'x", "b*'y", "b*'z",
            "c*'x", "c*'y", "c*'z"
        ]
        data = np.concatenate(
            (self.astar_prime, self.bstar_prime, self.cstar_prime)
            ).reshape(1,-1)
        
        return DataFrame(data=data, columns=columns)
    
    @property
    def info(self):
        print(f"""fitfile location "{self.filename}"
Peaks data comes from the corfile "{self.corfile}"
Created {self.timestamp}
Software used {self.software}

Material: {self.element}
Number of indexed spots: {self.number_indexed_spots}
Mean pixel deviation: {self.mean_pixel_deviation}

{" New lattice parameters ":=^28}
a': {self.new_lattice_parameters[0]:11.7f}
b': {self.new_lattice_parameters[1]:11.7f}
c': {self.new_lattice_parameters[2]:11.7f}
α': {self.new_lattice_parameters[3]:11.7f}
β': {self.new_lattice_parameters[4]:11.7f}
γ': {self.new_lattice_parameters[5]:11.7f}

{" Calibration parameters ":=^28}
distance: {self.CCDdict["DetectorParameters"][0]:7.2f} [mm]
x_center: {self.CCDdict["DetectorParameters"][1]:7.2f} [px]
y_center: {self.CCDdict["DetectorParameters"][2]:7.2f} [px]
    beta: {self.CCDdict["DetectorParameters"][3]:7.2f} [deg]
   gamma: {self.CCDdict["DetectorParameters"][4]:7.2f} [deg]
""")
    
    def plot(self):
        indexation_plot(self.peaklist)