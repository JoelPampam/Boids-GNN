import pygame  # loads Pygame Library
import sys     # loads pythons build in system tools
import random
import math
import csv
import struct

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Every simulation step (i.e. every unpaused frame), each boid's kinematic
# state and its "adjacency" info (which other boids fall within each
# behavior radius) is written as one row to a CSV file, and as one packed
# record to a binary file.
#
# Radii used to decide adjacency -- these match the distances already used
# inside Boid.separation / Boid.cohesion / Boid.alignment below.
SEPARATION_RADIUS = 35
COHESION_RADIUS = 100
ALIGNMENT_RADIUS = 100

CSV_PATH = "boids_log.csv"
BIN_PATH = "boids_log.bin"

CSV_FIELDS = [
    "step", "boid_id", "x", "y", "x_vel", "y_vel",
    "pre_planned", "separation", "cohesion", "alignment",
]

# ---------------------------------------------------------------------------
# Binary format (little-endian), one record per boid per step:
#
#   uint32   step
#   uint32   boid_id
#   float32  x
#   float32  y
#   float32  x_vel
#   float32  y_vel
#   uint8    pre_planned            (0 or 1)
#   uint16   separation_count
#   uint32[] separation_ids         (separation_count entries)
#   uint16   cohesion_count
#   uint32[] cohesion_ids
#   uint16   alignment_count
#   uint32[] alignment_ids
#
# There is no fixed record length (neighbor lists vary in size), so the
# file must be parsed sequentially from the start, reading each count
# before its id list.
# ---------------------------------------------------------------------------
RECORD_HEADER_FMT = "<IIffffB"  # step, boid_id, x, y, x_vel, y_vel, pre_planned


def pack_record(step, boid_id, x, y, x_vel, y_vel, pre_planned,
                separation_ids, cohesion_ids, alignment_ids):
    data = struct.pack(
        RECORD_HEADER_FMT,
        step, boid_id, x, y, x_vel, y_vel, 1 if pre_planned else 0,
    )
    for ids in (separation_ids, cohesion_ids, alignment_ids):
        data += struct.pack("<H", len(ids))
        if ids:
            data += struct.pack(f"<{len(ids)}I", *ids)
    return data


# Setup
pygame.init()  # starts pygame always needed

# creates a variable name screen and creates a window thats 900pixels wide and 650 in height
screen = pygame.display.set_mode((900, 650))
clock = pygame.time.Clock()

# names the window
pygame.display.set_caption("Boids Model")


class Boid:

    def __init__(self, boid_id):
        self.id = boid_id
        self.x = random.uniform(0, 900)
        self.y = random.uniform(0, 650)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.pre_planned = False  # not yet used, reserved for precomputed-path boids

    def cohesion(self, boids):
        center_x = 0
        center_y = 0
        count = 0

        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < 100:
                center_x += other.x
                center_y += other.y
                count += 1

        if count > 0:
            center_x /= count
            center_y /= count
            self.vx += (center_x - self.x) * 0.001
            self.vy += (center_y - self.y) * 0.001

    def separation(self, boids):
        move_x = 0
        move_y = 0

        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < 35:
                move_x -= dx
                move_y -= dy

        self.vx += move_x * 0.02
        self.vy += move_y * 0.02

    def limit_speed(self, max_speed):
        speed = (self.vx**2 + self.vy**2) ** 0.5    # total speed (distance formula)

        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed

    def alignment(self, boids):
        avg_vx = 0
        avg_vy = 0
        count = 0

        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < 100:
                avg_vx += other.vx
                avg_vy += other.vy
                count += 1

        if count > 0:
            avg_vx /= count
            avg_vy /= count
            self.vx += (avg_vx - self.vx) * 0.05
            self.vy += (avg_vy - self.vy) * 0.05

    def avoid_obstacles(self, obstacles):
        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            distance = (dx**2 + dy**2) ** 0.5
            danger_zone = radius + 30

            if distance < radius + 30:        # danger zone = radius + buffer
                if distance == 0:
                    distance = 0.1            # avoid dividing by zero
                strength = (danger_zone - distance) / danger_zone
                self.vx += (dx / distance) * strength * 2
                self.vy += (dy / distance) * strength * 2

    def enforce_no_overlap(self, obstacles):
        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            distance = (dx**2 + dy**2) ** 0.5

            if distance < radius and distance > 0:
                # push the boid to sit exactly on the edge of the obstacle
                self.x = ox + (dx / distance) * radius
                self.y = oy + (dy / distance) * radius

    def update(self, boids):
        self.cohesion(boids)
        self.separation(boids)
        self.alignment(boids)
        self.avoid_obstacles(obstacles)
        self.limit_speed(4)
        self.x += self.vx
        self.y += self.vy
        self.enforce_no_overlap(obstacles)

        if self.x > 900:
            self.x = 0
        if self.x < 0:
            self.x = 900
        if self.y > 650:
            self.y = 0
        if self.y < 0:
            self.y = 650

    def draw(self, screen):
        x, y = int(self.x), int(self.y)
        angle = math.atan2(self.vy, self.vx)   # direction of travel, in radians

        size = 10
        tip   = (x + math.cos(angle) * size,           y + math.sin(angle) * size)
        left  = (x + math.cos(angle + 2.5) * size*0.6, y + math.sin(angle + 2.5) * size*0.6)
        right = (x + math.cos(angle - 2.5) * size*0.6, y + math.sin(angle - 2.5) * size*0.6)

        pygame.draw.polygon(screen, (100, 200, 255), [tip, left, right])

    def neighbors_within(self, boids, radius):
        """Return a sorted list of boid_ids within `radius` of this boid."""
        ids = []
        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5
            if distance < radius:
                ids.append(other.id)
        ids.sort()
        return ids


def log_step(step, boids, csv_writer, bin_file):
    """Write one row per boid (CSV) and one packed record per boid (binary)."""
    for boid in boids:
        separation_ids = boid.neighbors_within(boids, SEPARATION_RADIUS)
        cohesion_ids = boid.neighbors_within(boids, COHESION_RADIUS)
        alignment_ids = boid.neighbors_within(boids, ALIGNMENT_RADIUS)

        csv_writer.writerow({
            "step": step,
            "boid_id": boid.id,
            "x": f"{boid.x:.4f}",
            "y": f"{boid.y:.4f}",
            "x_vel": f"{boid.vx:.4f}",
            "y_vel": f"{boid.vy:.4f}",
            "pre_planned": int(boid.pre_planned),
            "separation": "-".join(str(i) for i in separation_ids),
            "cohesion": "-".join(str(i) for i in cohesion_ids),
            "alignment": "-".join(str(i) for i in alignment_ids),
        })

        bin_file.write(pack_record(
            step, boid.id, boid.x, boid.y, boid.vx, boid.vy, boid.pre_planned,
            separation_ids, cohesion_ids, alignment_ids,
        ))


obstacles = []

# create 100 boids
boids = [Boid(i) for i in range(100)]
paused = False
step = 0

csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
csv_writer.writeheader()
bin_file = open(BIN_PATH, "wb")


def shutdown():
    csv_file.close()
    bin_file.close()
    pygame.quit()
    sys.exit()


# Game loop
# keeps looping everything in the while loop
while True:

    # collects a list of things that happen(mouse click, key press, closing the window)
    for event in pygame.event.get():

        # if the user clicks x close the window
        if event.type == pygame.QUIT:
            shutdown()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            if event.key == pygame.K_x:
                obstacles.clear()
            if event.key == pygame.K_ESCAPE:
                shutdown()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            if event.button == 1:                    # left click
                obstacles.append((mouse_x, mouse_y, 40))

            elif event.button == 3:                  # right click
                if obstacles:                         # only if there's at least one
                    closest = None
                    closest_dist = None

                    for obstacle in obstacles:
                        ox, oy, radius = obstacle
                        dist = ((ox - mouse_x)**2 + (oy - mouse_y)**2) ** 0.5

                        if closest_dist is None or dist < closest_dist:
                            closest = obstacle
                            closest_dist = dist

                    obstacles.remove(closest)

    # paints the entire window dark blue
    screen.fill((15, 20, 35))

    for ox, oy, radius in obstacles:
        pygame.draw.circle(screen, (200, 80, 80), (ox, oy), radius)

    if not paused:
        for boid in boids:
            boid.update(boids)

        log_step(step, boids, csv_writer, bin_file)
        step += 1

        if step % 60 == 0:      # flush to disk roughly once a second
            csv_file.flush()
            bin_file.flush()

    for boid in boids:
        boid.draw(screen)

    # shows everything done on the screen
    pygame.display.flip()
    clock.tick(60)