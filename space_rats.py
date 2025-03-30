import random
import numpy as np
import concurrent.futures
from collections import deque
import matplotlib.pyplot as plt

class SpaceRats:
    def __init__(self, D=30, alpha=0.1, space_rat_strategy='stationary'):
        self.D = D # Grid size
        self.alpha = alpha # Detector sensitivity
        self.grid = np.full((self.D, self.D), False)  # False is blocked, True is open
        self.open_positions = set() # Set of open positions
        self.move_offsets = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.num_turns = 0 # Number of turns taken by the bot
        self.space_rat_strategy = space_rat_strategy # Mobile or stationary space rat

    def get_neighbors(self, x, y):
        """Get four cardinal neighbors of a cell (x, y)"""
        return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

    def initialize_grid(self, placement=True):
        """Initialize the grid and positions of the bot and space rat"""
        # Step 1: Randomly select a start position and open it
        start_x, start_y = random.randint(1, self.D - 2), random.randint(1, self.D - 2) # Random start position within 28x28 grid
        self.grid[start_x, start_y] = True
        self.open_positions.add((start_x, start_y))

        # Step 2: Iteratively open cells with exactly one open neighbor
        self.iteratively_open_cells()

        # Step 3: Open half of the dead-end cells
        self.open_dead_ends()

        # Step 4: Initialize the weight grid for the space rat
        self.utility_grid = np.full((self.D, self.D), 0)
        for x, y in self.open_positions: # weight = 1 for open positions
            self.utility_grid[x, y] = 1

        # Step 5: Place bot and space rat
        if placement:
            self.place_bot_space_rat()

    def iteratively_open_cells(self):
        """Open cells iteratively that have exactly one open neighbor."""
        while True:
            options = []
            for x in range(1, self.D - 1):
                for y in range(1, self.D - 1):
                    if not self.grid[x, y]: # Closed cell
                        open_neighbors = sum(self.grid[nx, ny] for nx, ny in self.get_neighbors(x, y))
                        if open_neighbors == 1: # Cell with exactly one open neighbor
                            options.append((x, y))
            if not options: # No more cells to open
                break
            new_open_cell = random.choice(options)
            self.grid[new_open_cell] = True # Open the cell
            self.open_positions.add(new_open_cell)

    def open_dead_ends(self):
        """Open half of the dead-end cells with exactly one open neighbor."""
        for x in range(1, self.D - 1):
            for y in range(1, self.D - 1):
                if not self.grid[x, y]: # Closed cell
                    continue
                open_neighbors = sum(self.grid[neighbor] for neighbor in self.get_neighbors(x, y))
                # Open random cell with exactly one open neighbor with 50% probability
                if open_neighbors == 1 and random.random() < 0.5:
                    closed_neighbors = [(nx, ny) for nx, ny in self.get_neighbors(x, y) if not self.grid[nx, ny] and 0 < nx < self.D - 1 and 0 < ny < self.D - 1]
                    chosen = random.choice(closed_neighbors)
                    self.grid[chosen] = True # Open the cell
                    self.open_positions.add(chosen)

    def place_bot_space_rat(self):
        """Randomly place the bot and space rat"""
        open_cells_list = list(self.open_positions)
        self.bot_position = random.choice(open_cells_list)
        open_cells_list.remove(self.bot_position) # Ensure space rat is not placed at the bot position
        self.space_rat_position = random.choice(open_cells_list)
    
    def get_grid_string(self):
        """Get the string representation of the grid with bot and space rat positions"""
        grid_str = ""
        for x in range(self.D):
            for y in range(self.D):
                if (x, y) == self.bot_position:
                    grid_str += "B "
                elif (x, y) == self.space_rat_position:
                    grid_str += "R "
                elif self.grid[x, y]:
                    grid_str += ". "
                else:
                    grid_str += "# "
            grid_str += "\n"
        return grid_str
    
    def sense_blocked_cells(self, x, y):
        """Sense the number of blocked cells in the eight surrounding cells of (x, y)"""
        diagonal_cells = [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]
        surrounding_cells = self.get_neighbors(x, y) + diagonal_cells # Get all eight surrounding cells
        count = sum(1 for cell in surrounding_cells if not self.grid[cell]) # Count the number of blocked cells
        return count
    
    def use_space_rat_detector(self):
        """Use the space rat detector to detect the space rat"""
        manhattan_distance = abs(self.bot_position[0] - self.space_rat_position[0]) + abs(self.bot_position[1] - self.space_rat_position[1])
        if manhattan_distance == 0: # The bot is at the same position as the space rat
            return 'found'
        prob_detection = np.exp(-self.alpha * (manhattan_distance - 1)) # Probability of detection decreases with distance
        if random.random() < prob_detection:
            return 'detected'
        return 'not detected'

    def attempt_move(self, direction):
        """Attempt to move the bot in a direction"""
        if direction not in self.move_offsets.values(): # Not a valid move
            return False
        x, y = self.bot_position
        new_position = (x + direction[0], y + direction[1])
        if self.grid[new_position]: # Move the bot if the new position is open
            self.bot_position = new_position
            return True
        return False
    
    def move_space_rat(self):
        """Move the space rat to a random open direction"""
        open_neighbors = [(nx, ny) for nx, ny in self.get_neighbors(*self.space_rat_position) if self.grid[nx, ny]]
        self.space_rat_position = random.choice(open_neighbors)

    def identify_best_direction(self, options, curr_pos, explored, closed, num_blocked_cells):
        """Identify the best direction to move the bot"""

        def get_direction(curr_pos, direction_counts):
            """Get the best direction to move the bot based on the direction counts"""
            best_direction = max(direction_counts, key=direction_counts.get)
            next_position = (curr_pos[0] + best_direction[0], curr_pos[1] + best_direction[1])
            return best_direction, next_position

        # Get the possible positions based on the number of surrounding blocked cells
        possible_cells = []
        direction_counts = {key: 0 for key in self.move_offsets.values()}
        for cell in options:
            if self.sense_blocked_cells(cell[0], cell[1]) == num_blocked_cells: # Check if the number of blocked cells match
                possible_cells.append(cell)
                # Count the number of open cells in each direction
                for offset in self.move_offsets.values():
                    new_position = (cell[0] + offset[0], cell[1] + offset[1])
                    if self.grid[new_position]:
                        direction_counts[offset] += 1
        
        # Remove directions that have 0 count and pick the best direction
        direction_counts = {key: value for key, value in direction_counts.items() if value > 0}
        best_dir, next_pos = get_direction(curr_pos, direction_counts)

        # If the next position is closed, remove the direction
        while next_pos in closed:
            direction_counts.pop(best_dir)
            best_dir, next_pos = get_direction(curr_pos, direction_counts)
        # check if all directions are explored
        if all((curr_pos[0] + d[0], curr_pos[1] + d[1]) in explored for d in direction_counts.keys()):
            best_dir = random.choice(list(direction_counts.keys()))
            return possible_cells, best_dir
        # If the next position is explored, remove the direction
        while next_pos in explored and len(direction_counts) > 1: # Ensure there is at least one direction to move
            direction_counts.pop(best_dir)
            best_dir, next_pos = get_direction(curr_pos, direction_counts)
        return possible_cells, best_dir
    
    def update_possible_cells(self, possible_cells, direction, move_status):
        """Update the possible cells after the bot's attempted move"""
        dx, dy = direction
        # Filter the possible cells based on the move status
        possible_cells = [cell for cell in possible_cells if self.grid[cell[0] + dx, cell[1] + dy] == move_status]
        # Since the bot moved, update the possible cells to the new position
        if move_status:
            possible_cells = [(cell[0] + dx, cell[1] + dy) for cell in possible_cells]
        return possible_cells

    def update_utilities(self, ping_status, bot_position, bot_name):
        """Update the probabilities of the space rat position based on the ping status"""
        self.utility_grid[*bot_position] = 0 # Set the weight of the current position to 0
        for nx, ny in self.get_neighbors(*bot_position):
            if ping_status == 'detected': # Increase the weight of the neighbors
                if (nx, ny) not in self.open_positions: # Skip update for closed positions
                    continue
                # Skip update for bot2 if the number of pings is greater than 10 and the ping ratio is high
                if bot_name == 'bot2' and self.num_ping_attempts > 10:
                    ping_ratio = self.num_pings / self.num_ping_attempts
                    if ping_ratio > 0.9:
                        continue
                # Default increment for other cases
                if self.space_rat_strategy == 'mobile' or self.utility_grid[nx, ny] > 0:
                    self.utility_grid[nx, ny] += 1
            elif ping_status == 'not detected': # Set the weight of the neighbors to 0, as the space rat is not there
                self.utility_grid[nx, ny] = 0

    def update_relative_utilities(self, ping_status, bot_position, relative_weights):
        """Update the relative weights of the space rat position based on the ping status"""
        relative_weights[bot_position] = 0 # Set the weight of the current position to 0
        if ping_status == 'detected': # Increase the weight of the neighbors
            for neighbor in self.get_neighbors(*bot_position):
                if neighbor not in relative_weights:
                    relative_weights[neighbor] = 2 # Initialize the weight to 2 (1 more than the initial weight)
                elif self.space_rat_strategy == 'mobile' or relative_weights[neighbor] > 0:
                    relative_weights[neighbor] += 1
        elif ping_status == 'not detected': # Set the weight of the neighbors to 0, as the space rat is not there
            for neighbor in self.get_neighbors(*bot_position):
                relative_weights[neighbor] = 0

    def breadth_first_search(self, bot_position):
        """Find the shortest path from bot to a potential space rat location using BFS."""
        queue = deque([(bot_position, [])])
        visited = set([bot_position])
        goal = np.max(self.utility_grid) # Find the highest weight in the utility grid
        while queue:
            curr, path = queue.popleft()
            if self.utility_grid[*curr] == goal: # Found the closest cell with the highest weight
                while not path:
                    random_direction = random.choice(list(self.move_offsets.values()))
                    new_position = (curr[0] + random_direction[0], curr[1] + random_direction[1])
                    if self.grid[new_position]:
                        path.append(new_position)
                return path
            for neighbor in self.get_neighbors(*curr):
                # Add the neighbor to the queue if it is open and not visited
                if neighbor in self.open_positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def transfer_to_utility_grid(self, start, relative_weights):
        """Update the space rat weights based on the relative weights"""
        for key, value in relative_weights.items():
            cell = (start[0] + key[0], start[1] + key[1])
            if self.grid[cell]: # Update the weight only for open positions
                self.utility_grid[cell] = value

    def get_utility(self, location, visited, discount=0.6, depth=4):
        """Recursively calculate the expected utility of a location"""
        multiplier = 0.3 if location in visited else 1 # Reduce the weight for visited locations
        immediate_utility = self.utility_grid[*location]
        visited = visited + [location] # Add the current location to the visited set
        expected_future_utility = 0
        for neighbor in self.get_neighbors(*location): # Add neighbors' expected utility
            if neighbor in self.open_positions and neighbor not in visited and depth > 0:
                expected_future_utility += discount * self.get_utility(neighbor, visited, discount, depth - 1)
        return multiplier * (immediate_utility + expected_future_utility)

    def find_best_utility_move(self, bot_position, explored):
        """Find the shortest path from bot to a potential space rat location using A* search."""
        x, y = bot_position
        options = []
        for dx, dy in self.move_offsets.values(): # Calculate the utility for each direction
            nx, ny = x + dx, y + dy
            if self.grid[nx, ny]:
                utility = self.get_utility((nx, ny), explored)
                options.append(((dx, dy), utility))
        if any(option[1] > 0 for option in options): # Move to the highest utility cell if it is positive
            return max(options, key=lambda x: x[1])[0]
        else: # Backtrack using BFS to highest weight cell if all utilities are zero
            next_position = self.breadth_first_search(bot_position)[0]
            best_move = (next_position[0] - bot_position[0], next_position[1] - bot_position[1])
            return best_move

    def bot1_phase1(self):
        """Bot 1 Phase 1: Identify the bot position using the number of open surrouding cells and movement"""
        possible_cells = []
        curr_position = (0, 0) # Relative start position of the bot
        explored = [curr_position]
        closed = set()
        while True:
            num_blocked_cells = self.sense_blocked_cells(*self.bot_position) # Sense the number of blocked cells
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            possible_cells, best_direction = self.identify_best_direction(
                possible_cells if possible_cells else self.open_positions, # Start with open positions, then use shortlisted cells
                curr_position, explored, closed, num_blocked_cells)
            
            move_status = self.attempt_move(best_direction)
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            # add the current position to the explored set
            new_position = (curr_position[0] + best_direction[0], curr_position[1] + best_direction[1])
            if move_status:
                explored.append(new_position) # Add the relative position to the explored set
                curr_position = new_position
            else:
                closed.add(new_position) # Add the relative position to the closed set

            possible_cells = self.update_possible_cells(possible_cells, best_direction, move_status)
            if len(possible_cells) == 1: # Found the bot position
                return possible_cells[0]
    
    def bot1_phase2(self, bot_location):
        """Bot 2 Phase 2: Find the space rat using the space rat detector and movement"""
        while True:
            ping_status = self.use_space_rat_detector() # Use the space rat detector
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            if ping_status == 'found':
                break # End the game
            self.update_utilities(ping_status, bot_location, bot_name='bot1') # Update the space rat weights
            next_position = self.breadth_first_search(bot_location)[0] # Take the first step of the shortest path to the highest weight
            move = (next_position[0] - bot_location[0], next_position[1] - bot_location[1])
            self.attempt_move(move)
            bot_location = next_position
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
    
    def bot2_phase1(self):
        """Bot 2 Phase 1: Identify the bot position using the number of open surrouding cells and movement"""
        possible_cells = []
        curr_position = (0, 0) # Relative start position of the bot
        explored = [curr_position]
        closed = set()
        relative_weights = {} # Relative utility weights for the space rat
        self.num_pings, self.num_ping_attempts = 0, 0
        if self.collect_data:
            self.data = []
            possible_grid = np.full((self.D, self.D), 0)
            for x, y in self.open_positions:
                possible_grid[x, y] = 1
            self.data.append((possible_grid, self.utility_grid.copy(), self.num_turns))
        while True:
            ping_status = self.use_space_rat_detector() # Use the space rat detector
            self.num_turns += 1
            if ping_status == 'found':
                return curr_position, explored, True
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            self.update_relative_utilities(ping_status, curr_position, relative_weights)
            self.num_ping_attempts += 1
            if ping_status == 'detected':
                self.num_pings += 1
            if self.collect_data:
                self.data.append((possible_grid, self.utility_grid.copy(), self.num_turns))

            if len(possible_cells) == 1: # Found the bot position
                bot_position = possible_cells[0]
                # Calculate the start position based on the relative position and current bot position
                start = (bot_position[0] - curr_position[0], bot_position[1] - curr_position[1])
                self.transfer_to_utility_grid(start, relative_weights)
                explored = [(x + start[0], y + start[1]) for x, y in explored] # Update the explored set to match the full grid
                return possible_cells[0], explored, False

            num_blocked_cells = self.sense_blocked_cells(self.bot_position[0], self.bot_position[1])
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            possible_cells, best_direction = self.identify_best_direction(
                possible_cells if possible_cells else self.open_positions, # Start with open positions, then use shortlisted cells
                curr_position, explored, closed, num_blocked_cells)
            if self.collect_data:
                possible_grid = np.full((self.D, self.D), 0)
                for x, y in possible_cells:
                    possible_grid[x, y] = 1
                self.data.append((possible_grid, self.utility_grid.copy(), self.num_turns))
            
            move_status = self.attempt_move(best_direction)
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            # add the current position to the explored set
            new_position = (curr_position[0] + best_direction[0], curr_position[1] + best_direction[1])
            if move_status:
                explored.append(new_position) # Add the relative position to the explored set
                curr_position = new_position
            else:
                closed.add(new_position) # Add the relative position to the closed set
            possible_cells = self.update_possible_cells(possible_cells, best_direction, move_status)
            if self.collect_data:
                possible_grid = np.full((self.D, self.D), 0)
                for x, y in possible_cells:
                    possible_grid[x, y] = 1
                self.data.append((possible_grid, self.utility_grid.copy(), self.num_turns))

    def bot2_phase2(self, bot_location, explored):
        """Bot 2 Phase 2: Find the space rat using the space rat detector and movement"""
        while True:
            # Only update the utility if the bot has not visited the location before or the space rat is mobile
            if self.space_rat_strategy == 'mobile' or bot_location not in explored:
                ping_status = self.use_space_rat_detector() # Use the space rat detector
                self.num_turns += 1
                if ping_status == 'found':
                    break # End the game
                if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                    self.move_space_rat()
                self.num_ping_attempts += 1
                if ping_status == 'detected':
                    self.num_pings += 1
                self.update_utilities(ping_status, bot_location, bot_name='bot2') # Update the space rat weights
                explored.append(bot_location)
                if self.collect_data:
                    possible_grid = np.full((self.D, self.D), 0)
                    possible_grid[bot_location] = 1
                    self.data.append((possible_grid, self.utility_grid.copy(), self.num_turns))
            best_move = self.find_best_utility_move(bot_location, explored)
            next_position = (bot_location[0] + best_move[0], bot_location[1] + best_move[1])
            self.attempt_move(best_move)
            bot_location = next_position
            self.num_turns += 1
            if self.space_rat_strategy == 'mobile': # Move the space rat if it is mobile
                self.move_space_rat()
            if self.collect_data:
                possible_grid = np.full((self.D, self.D), 0)
                possible_grid[bot_location] = 1
                self.data.append((possible_grid, self.utility_grid.copy(), self.num_turns))

def apply_bot1_strategy(game):
    """Apply Bot 1's strategy to find the space rat"""
    grid_str = game.get_grid_string()
    bot_location = game.bot1_phase1() # Find the bot location
    game.bot1_phase2(bot_location) # Find the space rat
    grid_str += "Bot 1 Number of turns: " + str(game.num_turns) + "\n"
    print(grid_str)
    return game.num_turns

def apply_bot2_strategy(game):
    """Apply Bot 2's strategy to find the space rat"""
    grid_str = game.get_grid_string()
    bot_location, explored, space_rat_found = game.bot2_phase1() # Find the bot location
    if not space_rat_found: # Find the space rat if the space rat is not found in phase 1
        game.bot2_phase2(bot_location, explored)
    if game.collect_data:
        return game.data
    grid_str += "Bot 2 Number of turns: " + str(game.num_turns) + "\n"
    print(grid_str)
    return game.num_turns

def test_bots(D=30, alpha=0.1, iterations=100, space_rat_strategy='stationary', seed=520):
    """Test the two bots on the Space Rats game"""
    # Generate games with the same initial conditions for comparison
    bot1_games_list = [SpaceRats(D=D, alpha=alpha, space_rat_strategy=space_rat_strategy) for _ in range(iterations)]
    bot2_games_list = [SpaceRats(D=D, alpha=alpha, space_rat_strategy=space_rat_strategy) for _ in range(iterations)]
    random.seed(seed)
    for game in bot1_games_list:
        game.initialize_grid()
    random.seed(seed)
    for game in bot2_games_list:
        game.initialize_grid()

    # Parallelize game processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Get the number of turns taken by each bot
        bot1_turns = list(executor.map(apply_bot1_strategy, bot1_games_list))
        bot2_turns = list(executor.map(apply_bot2_strategy, bot2_games_list))
    return bot1_turns, bot2_turns

def plot_turn_counts(alpha_values, bot1_average_list, bot2_average_list):
    """Plot the average number of turns for different alpha values"""
    plt.plot(alpha_values, bot1_average_list, marker='.', color='r', label='Bot 1')
    plt.plot(alpha_values, bot2_average_list, marker='.', color='b', label='Bot 2')
    plt.xlabel('Alpha values')
    plt.ylabel('Average number of turns')
    plt.title('Average number of turns for different alpha values')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dim, iter = 30, 100 # Change the grid size and number of iterations as needed
    bot1_average_list, bot2_average_list = [], []
    # List of alpha values to test
    alpha_values = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                    0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]
    
    for alpha in alpha_values:
        print("Alpha value:", alpha)
        # Use space_rat_strategy='mobile'/'stationary' to test the bots with mobile or stationary space rat
        bot1_turns, bot2_turns = test_bots(D=dim, alpha=alpha, iterations=iter, space_rat_strategy='mobile')
        bot1_average_list.append(sum(bot1_turns) / iter)
        bot2_average_list.append(sum(bot2_turns) / iter)
    print("Bot 1 Average number of turns:", sum(bot1_average_list) / iter)
    print("Bot 2 Average number of turns:", sum(bot2_average_list) / iter)

    # Plot the average number of turns for different alpha values
    plot_turn_counts(alpha_values, bot1_average_list, bot2_average_list)