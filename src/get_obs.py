from src.utils import ObsHolder


obs_holder = ObsHolder.load("./saves/12")

obs_pos, obs_phse, obs_probs = obs_holder.get_og_obs()


print(obs_pos.tolist())
print()
print(obs_phse.tolist())
print()
print(obs_probs.tolist())

