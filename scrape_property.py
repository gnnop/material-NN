from mp_api.client import MPRester
import pickle
from prettyprint import prettyPrint

mpr = MPRester("Your API key here")

docs = mpr.materials.summary.search(fields=["ordering", "material_id", "energy_per_atom", "band_gap", "uncorrected_energy_per_atom", "structure"])
data = [
    {
                       "ordering": str(item.ordering),
                             "id": item.material_id,
                      "id_number": int("".join([ch for ch in item.material_id if ch.isdigit()])),
                "energy_per_atom": item.energy_per_atom,
                       "band_gap": item.band_gap,
    "uncorrected_energy_per_atom": item.uncorrected_energy_per_atom,
                      "structure": item.structure.to('POSCAR')
    }
    for item in docs
]

prettyPrint(data)

with open('scraped_properties.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)