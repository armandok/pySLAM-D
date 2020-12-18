import habitat_sim
from Config import Config

class Simulator:
    def __init__(self):
        test_scene = Config().scene

        sim_settings = {
            "width": 640,  # Spatial resolution of the observations
            "height": 480,
            "scene": test_scene,  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": True,  # RGB sensor
            "semantic_sensor": True,  # Semantic sensor
            "depth_sensor": True,  # Depth sensor
            "seed": 1,
        }

        cfg = self.make_cfg(sim_settings)
        self.sim = habitat_sim.Simulator(cfg)

        self.action_names = list(
            cfg.agents[
                sim_settings["default_agent"]
            ].action_space.keys()
        )

        self.total_frames = 0

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings["scene"]

        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.5)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.5)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def get_obs(self, action):
        observations = self.sim.step(action)
        rgb = observations["color_sensor"]
        # semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        self.total_frames += 1
        return rgb, depth

    def reset(self):

        agent = self.sim.get_agent(0)
        agent_state = agent.initial_state # agent.get_state()

        num_start_tries = 0
        while num_start_tries < 50:
            agent_state.position = self.sim.pathfinder.get_random_navigable_point()
            num_start_tries += 1
        agent.set_state(agent_state)
