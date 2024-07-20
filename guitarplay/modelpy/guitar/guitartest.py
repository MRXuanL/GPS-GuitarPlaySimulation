from absl.testing import absltest
from dm_control import mjcf
from guitarplay.modelpy.guitar import guitar
from guitarplay.modelpy.guitar import guitar_constants as cons



class GuitarTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        robot = guitar.Guitar()
        physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        for _ in range(100):
            physics.step()


    def test_keys(self) -> None:
        robot = guitar.Guitar()
        self.assertEqual(len(robot.keys_sites), cons.NUM_KEYS)
        for key in robot.keys_sites:
            self.assertEqual(key.tag, "site")
            
            
    def test_sensors(self) -> None:
        robot = guitar.Guitar()
        self.assertEqual(len(robot.keys_forces_sensors), cons.NUM_KEYS)
        for key in robot.keys_forces_sensors:
            self.assertEqual(key.tag, "touch")
        
        self.assertEqual(len(robot.keys_positions_sensors), cons.NUM_KEYS)
        for key in robot.keys_positions_sensors:
            self.assertEqual(key.tag, "framepos")

if __name__ == "__main__":
    absltest.main()
