from src.lp_model import OliveHarvestModel
from src.model_data import OliveHarvestData

data = OliveHarvestData.from_json("data/params.json")

model = OliveHarvestModel(data)
status = model.solve(msg=True)

print("Status:", status)
print("Objective:", model.objective_value())
print(model.get_x())