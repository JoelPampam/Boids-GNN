import pygame #loads Pygame Library
import sys #loads pythons build in system tools
import random
import math

# Setup
pygame.init() #starts pygame always needed

#creates a variable name screen and creates a window thats 900pixels wide and 650 in height
screen = pygame.display.set_mode((900, 650))
clock = pygame.time.Clock()

#names the window
pygame.display.set_caption("Boids Model")

class Boid:


    def __init__(self):
        self.x = random.uniform(0,900)
        self.y = random.uniform(0,650)
        self.vx = random.uniform(-2,2)
        self.vy = random.uniform(-2,2)

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
                strength = (danger_zone - distance)/danger_zone
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

obstacles = []

#create 20 boids
boids = [Boid() for _ in range(100)]
paused = False
# Game loop
#keeps looping everything in the while loop
while True:

    #collects a list of things that happen(mouse click, key press, closing the window)
    for event in pygame.event.get():

        #if the user clicks x close the window
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused      
            if event.key == pygame.K_x:
                obstacles.clear()
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


    #paints the entire window dark blue
    screen.fill((15, 20, 35))

    for ox, oy, radius in obstacles:
        pygame.draw.circle(screen, (200, 80, 80), (ox, oy), radius)
    
    if not paused:
        for boid in boids:
            boid.update(boids)

    for boid in boids:
        boid.draw(screen)
    
    #shows everything done on the screen
    pygame.display.flip()
    clock.tick(60)