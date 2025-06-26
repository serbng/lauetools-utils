import numpy as np

from laueutils.utils.strings import clean_string, remove_newline
character_list = ["[", "]", "\n", "#"]

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
    indexed_hkls: list[str] | should this be kept?
        fill_docstring      |
    
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
        self.filename = filename
        
        try:
            with open(filename, "r") as file:
                self._read_file_header(file)
                self._read_file_body(file)
                self._compute_reciprocal_space()
        except IOError:
            if verbose:
                print(f"Can't find the file \n{filename}")
        
    def _read_file_header(self, file):
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
        self.corfile = line.split(": ")[-1]
        
        line = remove_newline(file.readline())
        self.timestamp, self.software = line.lstrip("#File created at ").split(" with ")

    def _read_file_body(self, file):
        """Compare each read line to the dictionary and call the corresponding function to parse it"""
        # Dictionary whose keys are the file entries, and the values are the functions to read them
        file_entries = self._file_entries()
        
        line = file.readline()
        line = clean_string(line, ["\n", "#"])
        
        while line != "\n" and line != "":
            
            try:
                file_entries[line](file, line)
                
                line = file.readline()
                line = clean_string(line, ["\n", "#"])
                
            except KeyError:
                # for entries that have both text and data in the same line
                try:
                    line_beginning = line.split(":")[0]
                    file_entries[line_beginning](file, line)
                    
                    line = file.readline()
                    line = clean_string(line, ["\n", "#"])
                
                # if this doesn't work either, rip    
                except KeyError:
                    print(f"Could not read line: \n{line}")
                    
                    # moving on
                    line = file.readline()
                    line = clean_string(line, ["\n", "#"])
                    
    def _file_entries(self):
        return {# key: file entry to read
                # value: function to read such entry
        "Number of indexed spots": self._number_indexed_spots,
        "Element": self._element,
        "grainIndex": self._grain_index,
        "Mean Deviation(pixel)": self._mean_pixel_deviation,
        "spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev": self._peaks_data,
        "spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz": self._peaks_data,
        "spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz grainindex": self._peaks_data,
        "UB matrix in q= (UB) B0 G* ": self._UB,
        "B0 matrix in q= UB (B0) G*": self._B0,
        "UBB0 matrix in q= (UB B0) G* i.e. recip. basis vectors are columns in LT frame: astar = UBB0[:,0], bstar = UBB0[:,1], cstar = UBB0[:,2]. (abcstar as columns on xyzlab1, xlab1 = ui, ui = unit vector along incident beam)": self._UBB0,
        "Euler angles phi theta psi (deg)": self._euler_angles,
        "deviatoric strain in direct crystal frame (10-3 unit)": self._deviatoric_strain_crystal_frame,
        "deviatoric strain in sample2 frame (10-3 unit)": self._deviatoric_strain_sample_frame,
        "new lattice parameters": self._new_lattice_parameters,
        "CCDLabel": self._camera_dict,
        }
    
    def _number_indexed_spots(self, file, line):
        # Sample entry:
        # "#Number of indexed spots: 185"
        self.number_indexed_spots = int(line.split()[-1])
    
    def _element(self, file, line):
        # Sample entry:
        # "#Element"
        # "#4H-SiC"
        line = clean_string(file.readline(), character_list)
        self.element = line
        
    def _grain_index(self, file, line):
        line = clean_string(file.readline(), character_list)
        # throw it in the trash because I don't really want it
    
    def _mean_pixel_deviation(self, file, line):
        # Sample entry:
        # "#Mean Deviation(pixel): 0.174"
        self.mean_pixel_deviation = float(line.split()[-1])
        
    def _peaks_data(self, file, line):
        # sample entry:
        # "##spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev"
        # "0.000000 61195.040000 0.000000 0.000000 6.000000 85.780427 0.861945 1023.590000 1196.750000 5.431640 0.000000 0.086716"
        self.peak = {}
        
        for peak_number in range(self.number_indexed_spots):
            line = file.readline().split()
            
            value   = [float(element) for element in line]
            h, k, l = [int(value[i]) for i in range(2,5)]
            key = f"{h} {k} {l}"
            
            self.peak[key] = np.array(value)
            
        self.indexed_hkls_list = list(self.peak.keys())
    
    def _UB(self, file, line):
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
        
        self.UB = np.array(UB)
        
    def _B0(self, file, line):
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
        
        self.B0 = np.array(B0)
    
    def _UBB0(self, file, line):
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
        
        self.UBB0 = np.array(UBB0)
    
    def _euler_angles(self, file, line):
        # sample entry:
        #
        # "#Euler angles phi theta psi (deg)"
        # "#[298.044  42.864  89.072]"
        
        line = clean_string(file.readline(), character_list).split()
        
        self.euler_angles = np.array([float(element) for element in line])
    
    def _deviatoric_strain_crystal_frame(self, file, line):
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
        
        self.deviatoric_strain_crystal_frame = np.array(dev_crystal)
    
    def _deviatoric_strain_sample_frame(self, file, line):
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
        
        self.deviatoric_strain_sample_frame = np.array(dev_sample)
        
    def _new_lattice_parameters(self, file, line):
        # sample entry:
        # "#new lattice parameters"
        # "#[  3.073       3.0727618  10.0603084  90.0125899  90.0198677 120.0106641]"
        line = clean_string(file.readline(), character_list).split()
        
        self.new_lattice_parameters = np.array([float(element) for element in line])
        
    def _camera_dict(self, file, line):
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
        
        self.CCDdict = camera_dict
            
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

    def print_indexed_hkls(self, nb_cols: int = 6):

        for element_number, indices in enumerate(self.indexed_hkls_list):
            #sample line: '0 0 -6'
            #split line to get h, k and l -> convert string to int
            h, k, l = [int(index) for index in indices.split()]
            
            print(f"[{h:3d}, {k:3d}, {l:3d}] ", end = "")
            if element_number!=0 and element_number % (nb_cols-1) == 0:
                print()