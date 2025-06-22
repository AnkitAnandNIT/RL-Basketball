import pygame
import numpy as np
import sys

WIDTH, HEIGHT = 600, 400
PLAYER_SIZE = 20
BALL_RADIUS = 10


class BasketballEnv:
    def __init__(self):
        pygame.init()
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("RL Football Game")
        self.reset()

    def reset(self):
        self.red = pygame.Rect(50, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
        self.blue = pygame.Rect(WIDTH - 70, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
        self.ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_RADIUS * 2, BALL_RADIUS * 2)
        self.ball_vel = [0.0, 0.0]  # [vx, vy]
        goal_width = 60
        goal_height = 350

        # Assuming red team scores on the right (blue goal), and blue scores on left (red goal)
        self.goal_red = pygame.Rect(0, (HEIGHT - goal_height) // 2, 10, goal_height)  # Left side
        self.goal_blue = pygame.Rect(WIDTH - 10, (HEIGHT - goal_height) // 2, 10, goal_height)  # Right side

        self.prev_ball_position = pygame.Vector2(self.ball.center)
        self.ball_velocity = pygame.Vector2(0, 0)
        return self.get_state()

    def get_state(self):
        state = [
            self.red.x / WIDTH, self.red.y / HEIGHT,
            self.blue.x / WIDTH, self.blue.y / HEIGHT,
            self.ball.x / WIDTH, self.ball.y / HEIGHT,
            self.ball_vel[0] / 10.0, self.ball_vel[1] / 10.0
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action_red, action_blue):
        self._move(self.red, action_red)
        self._move(self.blue, action_blue)

        dist_red = self._distance(self.red.center, self.ball.center)
        dist_blue = self._distance(self.blue.center, self.ball.center)

        # Negative reward proportional to distance from ball
        reward_red = -10 * (dist_red / WIDTH)
        reward_blue = -10 * (dist_blue / WIDTH)

        # Reward for moving toward the ball
        red_vec = np.array(self.ball.center) - np.array(self.red.center)
        blue_vec = np.array(self.ball.center) - np.array(self.blue.center)
        red_move = np.array([
            self.red.x - getattr(self, 'prev_red_x', self.red.x),
            self.red.y - getattr(self, 'prev_red_y', self.red.y)
        ])
        blue_move = np.array([
            self.blue.x - getattr(self, 'prev_blue_x', self.blue.x),
            self.blue.y - getattr(self, 'prev_blue_y', self.blue.y)
        ])

        if np.linalg.norm(red_vec) > 0:
            reward_red += 0.8 * (
                np.dot(red_vec, red_move) / (np.linalg.norm(red_vec) * max(np.linalg.norm(red_move), 1e-5))
            )
        if np.linalg.norm(blue_vec) > 0:
            reward_blue += 0.8 * (
                np.dot(blue_vec, blue_move) / (np.linalg.norm(blue_vec) * max(np.linalg.norm(blue_move), 1e-5))
            )

        # Save previous positions for next step
        self.prev_red_x, self.prev_red_y = self.red.x, self.red.y
        self.prev_blue_x, self.prev_blue_y = self.blue.x, self.blue.y

        # Handle collisions (reward for hitting)
        red_hit = self._handle_collision(self.red)
        blue_hit = self._handle_collision(self.blue)

        if red_hit:
            reward_red += 100
        if blue_hit:
            reward_blue += 100


        # Update ball movement
        self.ball.x += int(self.ball_vel[0])
        self.ball.y += int(self.ball_vel[1])
        self.ball_vel[0] *= 0.95
        self.ball_vel[1] *= 0.95
        self.ball.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

        if self.ball.left <= 0 or self.ball.right >= WIDTH:
            self.ball_vel[0] *= -1
        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_vel[1] *= -1

        # Calculate new velocity
        new_position = pygame.Vector2(self.ball.center)
        self.ball_velocity = new_position - self.prev_ball_position
        self.prev_ball_position = new_position

        # Direction to opponent goal from ball
        to_blue_goal = pygame.Vector2(self.goal_blue.center) - pygame.Vector2(self.ball.center)
        to_red_goal = pygame.Vector2(self.goal_red.center) - pygame.Vector2(self.ball.center)

        # Normalize directions (avoid zero-length)
        if to_blue_goal.length() != 0:
            to_blue_goal = to_blue_goal.normalize()
        if to_red_goal.length() != 0:
            to_red_goal = to_red_goal.normalize()

        # Use dot product to measure alignment of ball movement with goal direction
        dot_red = self.ball_velocity.dot(to_blue_goal)  # red pushing toward blue goal
        dot_blue = self.ball_velocity.dot(to_red_goal)  # blue pushing toward red goal

        # Reward if moving ball in correct direction
        reward_red += max(0, dot_red) * 0.2
        reward_blue += max(0, dot_blue) * 0.2


        # Goal detection
        done = False
        if self.ball.left <= 0:
            reward_red -= 10
            reward_blue += 10000
            done = True
        elif self.ball.right >= WIDTH:
            reward_red += 10000
            reward_blue -= 10
            done = True

        return self.get_state(), reward_red, reward_blue, done


    def _move(self, player, action):
        speed = 5
        dx, dy = 0, 0
        if action == 0: dy = -speed
        elif action == 1: dy = speed
        elif action == 2: dx = -speed
        elif action == 3: dx = speed
        player.move_ip(dx, dy)
        player.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

    def _handle_collision(self, player):
        if self.ball.colliderect(player):
            dx = self.ball.centerx - player.centerx
            dy = self.ball.centery - player.centery
            dist = max((dx**2 + dy**2) ** 0.5, 1e-5)
            push_strength = 7.0
            self.ball_vel[0] = (dx / dist) * push_strength
            self.ball_vel[1] = (dy / dist) * push_strength
            return True
        return False

    def _distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    def render(self, episode=0, score_red=0, score_blue=0):
        self.win.fill((255, 255, 255))
        #draw player and ball
        pygame.draw.rect(self.win, (255, 0, 0), self.goal_red)    # Red goal (left)
        pygame.draw.rect(self.win, (0, 0, 255), self.goal_blue)   # Blue goal (right)

        pygame.draw.line(self.win, (200, 200, 200), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
        pygame.draw.circle(self.win, (200, 200, 200), (300,200), 100, width=2)
        pygame.draw.rect(self.win, (255, 0, 0), self.red)
        pygame.draw.rect(self.win, (0, 0, 255), self.blue)
        pygame.draw.ellipse(self.win, (0, 0, 0), self.ball)

        #text
        font = pygame.font.SysFont(None, 24)
        info = font.render(f"Episode: {episode} | Red: {score_red}  Blue: {score_blue}", True, (0, 0, 0))
        self.win.blit(info, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

