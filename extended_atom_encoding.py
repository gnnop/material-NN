import mendeleev
import math
from prettyprint import prettyPrint
import pickle

periodic_table_coordinates = [
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 6, 0],
    [1, 6, 1],
    [1, 7, 0],
    [1, 7, 1],
    [1, 8, 0],
    [1, 8, 1],
    [2, 0, 0],
    [2, 0, 1],
    [2, 6, 0],
    [2, 6, 1],
    [2, 7, 0],
    [2, 7, 1],
    [2, 8, 0],
    [2, 8, 1],
    [3, 0, 0],
    [3, 0, 1],
    [3, 1, 0],
    [3, 1, 1],
    [3, 2, 0],
    [3, 2, 1],
    [3, 3, 0],
    [3, 3, 1],
    [3, 4, 0],
    [3, 4, 1],
    [3, 5, 0],
    [3, 5, 1],
    [3, 6, 0],
    [3, 6, 1],
    [3, 7, 0],
    [3, 7, 1],
    [3, 8, 0],
    [3, 8, 1],
    [4, 0, 0],
    [4, 0, 1],
    [4, 1, 0],
    [4, 1, 1],
    [4, 2, 0],
    [4, 2, 1],
    [4, 3, 0],
    [4, 3, 1],
    [4, 4, 0],
    [4, 4, 1],
    [4, 5, 0],
    [4, 5, 1],
    [4, 6, 0],
    [4, 6, 1],
    [4, 7, 0],
    [4, 7, 1],
    [4, 8, 0],
    [4, 8, 1],
    [5, 0, 0],
    [5, 0, 1],
    [5, 1, 0],
    [5, 1, 1],
    [5, 2, 0],
    [5, 2, 1],
    [5, 3, 0],
    [5, 3, 1],
    [5, 4, 0],
    [5, 4, 1],
    [5, 5, 0],
    [5, 5, 1],
    [5, 6, 0],
    [5, 6, 1],
    [5, 7, 0],
    [5, 7, 1],
    [5, 8, 0],
    [5, 8, 1],
    [6, 0, 0],
    [6, 0, 1],
    [6, 1, 0],
    [6, 1, 1],
    [6, 2, 0],
    [6, 2, 1],
    [6, 3, 0],
    [6, 3, 1],
    [6, 4, 0],
    [6, 4, 1],
    [6, 5, 0],
    [6, 5, 1],
    [6, 6, 0],
    [6, 6, 1],
    [6, 7, 0],
    [6, 7, 1],
    [6, 8, 0],
    [6, 8, 1],
    [5, 9, 0],
    [5, 9, 1],
    [5, 10, 0],
    [5, 10, 1],
    [5, 11, 0],
    [5, 11, 1],
    [5, 12, 0],
    [5, 12, 1],
    [5, 13, 0],
    [5, 13, 1],
    [5, 14, 0],
    [5, 14, 1],
    [5, 15, 0],
    [5, 15, 1],
    [6, 9, 0],
    [6, 9, 1],
    [6, 10, 0],
    [6, 10, 1],
    [6, 11, 0],
    [6, 11, 1],
    [6, 12, 0],
    [6, 12, 1],
    [6, 13, 0],
    [6, 13, 1],
    [6, 14, 0],
    [6, 14, 1],
    [6, 15, 0],
    [6, 15, 1],
]


def flatten(nested_list):
    """Flattens a list of lists of arbitrary depth. Convert to a list with list(flatten(item))"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def get_numeric_property_of_all_atoms(property_name) -> list:
    return [vars(mendeleev.element(i))[property_name] or 0 for i in range(1,119)]

def numeric_property_to_normalized_values(numeric_property):
    '''
    Makes some numeric properties easier to digest for neural networks with Fourier Features
    '''
    maxValue = max([abs(p) for p in numeric_property])
    fourierFeaturesPeriodScale = 2
    absoluteCoordinateLength = math.log(maxValue) / math.log(fourierFeaturesPeriodScale)          # log base 2 of the largest value
    absoluteCoordinateLength = absoluteCoordinateLength + 5        # plus 5 to get 5 decimal places
    processed_values = [{
        "iszero":     p==0,
        "sign":       math.copysign(1, p),
        "normalized": p/maxValue,
        "fourierfeatures": [
            math.cos(
                math.tau
                * fourierFeaturesPeriodScale**(i+1)
                / maxValue
                * p
            ) 
            for i in range(int(absoluteCoordinateLength))
        ]
    } for p in numeric_property]
    return [
        list(flatten(pv.values()))
        for pv in processed_values
    ]

def one_hot_encode(items):
    "Performs one-hot encoding on a list of items"
    unique_items = list(set(items))
    one_hot_encoded_list = []
    vector_length = len(unique_items)
    for item in items:
        vector = [0] * vector_length
        index = unique_items.index(item)
        vector[index] = 1
        one_hot_encoded_list.append(vector)
    return one_hot_encoded_list


def generate_extended_atom_encodings_db():

    # Group needs processing
    groups = get_numeric_property_of_all_atoms("group")
    groups = [g.symbol if g else 0 for g in groups]
    coords_row, coords_col, coords_spin = zip(*periodic_table_coordinates)



    db = {
        "exists":                [1]*len(groups),
        "coords_row":            one_hot_encode(coords_row),
        "coords_col":            one_hot_encode(coords_col),
        "coords_spin":           one_hot_encode(coords_spin),
        # "atomic_radius":         numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("atomic_radius")),
        # "abundance_crust":       numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("abundance_crust")),
        # "abundance_sea":         numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("abundance_sea")),
        # "atomic_radius_rahm":    numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("atomic_radius_rahm")),
        # "atomic_volume":         numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("atomic_volume")),
        # "atomic_weight":         numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("atomic_weight")),
        # "block":                 one_hot_encode(get_numeric_property_of_all_atoms("block")),
        # "dipole_polarizability": numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("dipole_polarizability")),
        # "electron_affinity":     numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("electron_affinity")),
        # "en_ghosh":              numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("en_ghosh")),
        # "en_pauling":            numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("en_pauling")),
        # "econf":                 one_hot_encode(get_numeric_property_of_all_atoms("econf")),
        # "group":                 one_hot_encode(groups),
        # "is_monoisotopic":       one_hot_encode(get_numeric_property_of_all_atoms("is_monoisotopic")),
        # "is_radioactive":        one_hot_encode(get_numeric_property_of_all_atoms("is_radioactive")),
        # "lattice_constant":      numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("lattice_constant")),
        # "lattice_structure":     one_hot_encode(get_numeric_property_of_all_atoms("lattice_structure")),
        # "metallic_radius":       numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("metallic_radius")),
        # "metallic_radius_c12":   numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("metallic_radius_c12")),
        # "vdw_radius":            numeric_property_to_normalized_values(get_numeric_property_of_all_atoms("vdw_radius")),
    }
    prettyPrint([entry[0] for entry in db.values()])
    atoms = [
        list(flatten([entry[element_index] for entry in db.values()]))
        for element_index in range(118)
    ]
    prettyPrint(atoms)
    return atoms

# The database
db = None
cache_filename = "extended_atom_encoding.pickle"
try:
    with open(cache_filename, 'rb') as handle:
        db = pickle.load(handle)
except:
    print("extended_atom_encoding: No cache found. Generating database")
    db = generate_extended_atom_encodings_db()
    try:
        with open(cache_filename, 'wb') as handle:
            pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("Failed to save atom database")
    # end except
# end except

def get_atom_from_db(atom_name):
    return db[mendeleev.element(atom_name).protons-1]
# end get_atom_from_db