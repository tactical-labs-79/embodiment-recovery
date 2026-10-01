import numpy as np

import robosuite as suite

ROBOT_TYPE = "Panda"

ENVIRONMENT = "Lift"

STEP_SIZE = 0.02

KEY_ACTIONS = {

    "w": np.array([1, 0, 0, 0, 0, 0, 0]),

    "s": np.array([-1, 0, 0, 0, 0, 0, 0]),

    "a": np.array([0, -1, 0, 0, 0, 0, 0]),

    "d": np.array([0, 1, 0, 0, 0, 0, 0]),

    "e": np.array([0, 0, 1, 0, 0, 0, 0]),

    "q": np.array([0, 0, -1, 0, 0, 0, 0]),

}

KEY_LABELS = {

    "w": "MAJU", "s": "MUNDUR", "a": "KIRI", "d": "KANAN",

    "e": "NAIK", "q": "TURUN",

}

class KeyboardController:

    def __init__(self):

        from pynput import keyboard

        self.pressed = set()

        self.grab = 0.0

        self.should_reset = False

        self.should_quit = False

        self.listener = keyboard.Listener(

            on_press=self._on_press, on_release=self._on_release

        )

        self.listener.start()

    def _on_press(self, key):

        from pynput import keyboard

        try:

            k = key.char.lower()

            if k in KEY_ACTIONS:

                self.pressed.add(k)

            elif k == "r":

                self.should_reset = True

        except AttributeError:

            if key == keyboard.Key.space:

                self.grab = 1.0

            elif key == keyboard.Key.shift:

                self.grab = -1.0

            elif key == keyboard.Key.esc:

                self.should_quit = True

    def _on_release(self, key):

        from pynput import keyboard

        try:

            k = key.char.lower()

            if k in self.pressed:

                self.pressed.discard(k)

        except AttributeError:

            if key in (keyboard.Key.space, keyboard.Key.shift):

                self.grab = 0.0

    def get_action(self) -> np.ndarray:

        action = np.zeros(7)

        for k in self.pressed:

            action[:6] += KEY_ACTIONS[k][:6]

        action[6] = self.grab

        action[:3] *= STEP_SIZE

        return action

    def active_labels(self) -> str:

        labels = [KEY_LABELS[k] for k in self.pressed]

        if self.grab > 0:

            labels.append("GRAB")

        elif self.grab < 0:

            labels.append("RELEASE")

        return " + ".join(labels) if labels else "diam"

    def stop(self):

        self.listener.stop()

def banner():

    print("=" * 60)

    print("  ROBOT KEYBOARD CONTROL")

    print("=" * 60)

    print("  W / S      : maju / mundur")

    print("  A / D      : kiri / kanan")

    print("  Q / E      : turun / naik")

    print("  SPASI      : jepit (grab)")

    print("  SHIFT      : lepas jepitan")

    print("  R          : reset posisi awal")

    print("  ESC        : keluar")

    print("=" * 60)

    print("  (klik jendela simulasi 3D dulu biar keyboard kebaca)")

    print("=" * 60)

def main():

    banner()

    print("[Robot] Menyalakan simulasi...")

    env = suite.make(

        env_name=ENVIRONMENT,

        robots=ROBOT_TYPE,

        has_renderer=True,

        has_offscreen_renderer=False,

        use_camera_obs=False,

        control_freq=20,

    )

    env.reset()

    print("[Robot] Simulasi siap. Robot berdiri di posisi awal.")

    controller = KeyboardController()

    last_label = ""

    try:

        while True:

            if controller.should_quit:

                print("[Robot] Keluar.")

                break

            if controller.should_reset:

                env.reset()

                controller.should_reset = False

                print("[Robot] >> Posisi direset ke awal.")

            action = controller.get_action()

            env.step(action)

            env.render()

            label = controller.active_labels()

            if label != last_label:

                print(f"[Robot] Aksi: {label}")

                last_label = label

    except KeyboardInterrupt:

        print("\n[Robot] Dihentikan (Ctrl+C).")

    finally:

        controller.stop()

        env.close()

        print("[Robot] Selesai.")

if __name__ == "__main__":

    main()
