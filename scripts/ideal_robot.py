# %% [markdown]
# ## 3.2 ロボットの動き
#

# %% [markdown]
# ### 3.2.1 世界座標系と描画
#

# %%
# %matplotlib ipympl

import math

import matplotlib.animation as anm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# %%
class World:
    def __init__(self, time_span: float, time_interval: float, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        elems = []

        if self.debug:
            for i in range(10):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig,
                self.one_step,
                fargs=(elems, ax),
                frames=int(self.time_span / self.time_interval) + 1,
                interval=int(self.time_interval * 1000),
                repeat=False,
            )
            plt.show()

    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)


# %%
class Agent:
    def __init__(self, nu, omega) -> None:
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


# %%
class IdealRobot:
    def __init__(self, pose, agent: Agent = None, sensor=None, color="black") -> None:
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor

    def draw(self, ax, elems):
        x, y, theta = self.pose

        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color=self.color)

        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot(
            [e[0] for e in self.poses],
            [e[1] for e in self.poses],
            linewidth=0.5,
            color="black",
        )
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu * math.cos(t0), nu * math.sin(t0), omega]) * time
        else:
            return pose + np.array(
                [
                    nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)),
                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                    omega * time,
                ]
            )


# %%
class Landmark:
    def __init__(self, x, y) -> None:
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems: list):
        c = ax.scatter(
            self.pos[0],
            self.pos[1],
            s=100,
            marker="*",
            label="landmarks",
            color="orange",
        )
        elems.append(c)
        elems.append(
            ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10)
        )


# %%
class Map:
    def __init__(self) -> None:
        self.landmarks = []

    def append_landmark(self, landmark: Landmark) -> None:
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems: list) -> None:
        for lm in self.landmarks:
            lm.draw(ax, elems)


# %%
class IdealCamera:
    def __init__(
        self,
        env_map,
        distance_range=(0.5, 6.0),
        direction_range=(-math.pi / 3, math.pi / 3),
    ) -> None:
        self.map: Map = env_map
        self.last_data: list = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polar_pos):
        if polar_pos is None:
            return False

        return (
            self.distance_range[0] <= polar_pos[0] <= self.distance_range[1]
            and self.direction_range[0] <= polar_pos[1] <= self.direction_range[1]
        )

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            if self.visible(p):
                observed.append((p, lm.id))
        self.last_data = observed
        return observed

    def draw(self, ax, elems, cam_pose):
        for lm in self.last_data:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi:
            phi -= 2 * np.pi
        while phi < -np.pi:
            phi += 2 * np.pi
        return np.array([np.hypot(*diff), phi]).T


# %%
if __name__ == "__main__":
    world = World(36, 0.1)
    print("aaa")

    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0 / 180 * math.pi)

    robot1 = IdealRobot(
        np.array([2, 3, math.pi / 6]).T, agent=straight, sensor=IdealCamera(m)
    )
    robot2 = IdealRobot(
        np.array([-2, -1, math.pi / 5 * 6]).T,
        agent=circling,
        sensor=IdealCamera(m),
        color="red",
    )
    # robot3 = IdealRobot(np.array([0, 0, 0]).T, color="blue")

    world.append(robot1)
    world.append(robot2)
    # world.append(robot3)

    world.draw()
