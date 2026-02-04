from optimization_model.lp_model import OliveHarvestModel, OliveHarvestModel_ServiceLevel
from optimization_model.model_data import OliveHarvestData, OliveHarvestDataFromJSON


data = OliveHarvestData.from_json("data/params.json")

# get_data weather 

# train the diffusion model to generate weather scenarios

# generate weather scenarios

# get mean and std for uncertain parameters 

json_path = "/mnt/data/params_model.json"

data = OliveHarvestDataFromJSON(json_path=json_path, alpha=0.95)
model = OliveHarvestModel_ServiceLevel(data)

status = model.solve(msg=True)
print("Status:", status)
print("Objective (oil):", model.objective_value())
print("ReqBudget:", model.required_budget_value())
print("Nonzero x:", len(model.get_x()))
    