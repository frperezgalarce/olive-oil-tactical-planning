from src.lp_model import OliveHarvestModel
from src.model_data import OliveHarvestData

data = OliveHarvestData.from_json("data/params.json")

# get_data weather 

# train the diffusion model to generate weather scenarios

# generate weather scenarios

# get mean and std for uncertain parameters 

model = OliveHarvestModel(data)

status = model.solve(msg=True)

print("Status:", status)
print("Objective:", model.objective_value())
print(model.get_x())