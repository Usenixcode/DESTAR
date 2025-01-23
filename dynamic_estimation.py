class CarlaRLModelPredictor:
    def __init__(self, model_path, model):
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  
        self.predicted_positions = []  
        self.actual_positions = []  
        self.first_step = True  
        self.prediction_step = 0  
        self.previous_predicted_x = None  
        self.previous_predicted_y = None  
        self.previous_actual_x = None  
        self.previous_actual_y = None  
        self.sequence_length = 10  
        self.node_features_seq = []  
        self.adj_matrix_seq = []  
        self.edge_attr_seq = []  

    def get_ego_vehicle_features(self, vehicle):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        angular_velocity = vehicle.get_angular_velocity()
        acceleration = vehicle.get_acceleration()

        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw
        vx = velocity.x
        vy = velocity.y
        ax = vy * yaw + (1 / m) * (
        (Fx_FL + Fx_FR) * np.cos(control.steer) -
        (Fy_FL + Fy_FR) * np.sin(control.steer) +
        Fx_RL + Fx_RR - CA * vx**2
        )

        # Calculate lateral acceleration ay
        ay = -vx * yaw + (1 / m) * (
            (Fx_FL + Fx_FR) * np.sin(control.steer) +
            (Fy_FL + Fy_FR) * np.cos(control.steer) +
            Fy_RL + Fy_RR
        )

        # Calculate angular acceleration alpha
        yaw_rate = (1 / J) * (
            lf * (
                (Fx_FL + Fx_FR) * np.sin(control.steer) +
                (Fy_FL + Fy_FR) * np.cos(control.steer)
            ) - lr * (Fy_RL + Fy_RR)
        )

        return np.array([x, y, yaw, vx, vy, yaw_rate, ax, ay])

    def calculate_edge_attributes(self, nodes, max_distance=10, max_neighbors=5):
        
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        edge_attributes = {}

        for i in range(num_nodes):
            node_i = nodes[i]
            distances = []
            for j in range(num_nodes):
                if i == j:
                    continue
                node_j = nodes[j]
                relative_position = np.array([node_j[0] - node_i[0], node_j[1] - node_i[1]])
                distance = np.linalg.norm(relative_position)
                if distance <= max_distance:
                    distances.append((distance, j))

            distances.sort()
            neighbors = distances[:max_neighbors]

            for dist, j in neighbors:
                adj_matrix[i, j] = 1
                node_j = nodes[j]
                relative_velocity = np.array([node_j[3] - node_i[3], node_j[4] - node_i[4]])
                relative_acceleration = np.array([node_j[6] - node_i[6], node_j[7] - node_i[7]])
                edge_attributes[(i, j)] = {
                    'relative_position': relative_position,
                    'relative_velocity': relative_velocity,
                    'relative_acceleration': relative_acceleration
                }

        return adj_matrix, edge_attributes


    def save_predictions(self, filename="predictions.csv"):
        """Save predicted and actual positions to a CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Predicted_X", "Predicted_Y", "Actual_X", "Actual_Y"])
            for step, (pred, actual) in enumerate(zip(self.predicted_positions, self.actual_positions), start=1):
                writer.writerow([step, pred[0], pred[1], actual[0], actual[1]])
        print(f"Predictions saved to {filename}")
    def predict_next_action(self, ego_vehicle, other_vehicles, dt=0.05, wheelbase=4.8):
        # Step 1: Get ego and other vehicle features
        ego_vehicle_features = self.get_ego_vehicle_features(ego_vehicle)
        other_vehicle_features = [self.get_ego_vehicle_features(vehicle) for vehicle in other_vehicles]
        node_features = [ego_vehicle_features] + other_vehicle_features

        # Step 2: Update GAT-LSTM sequence inputs
        adj_matrix, edge_attributes = self.calculate_edge_attributes(node_features)
        self.node_features_seq.append(node_features)
        self.adj_matrix_seq.append(adj_matrix)
        self.edge_attr_seq.append(edge_attributes)
        
        if len(self.node_features_seq) > self.sequence_length:
            self.node_features_seq.pop(0)
            self.adj_matrix_seq.pop(0)
            self.edge_attr_seq.pop(0)
        
        if len(self.node_features_seq) < self.sequence_length:
            return  

        # Step 3: Prepare model inputs
        
        node_features_numpy = np.array(self.node_features_seq)  
        adj_matrix_numpy = np.array(self.adj_matrix_seq)  

       
        num_nodes = len(node_features)  
        num_edge_features = 6  
        edge_attr_array = np.zeros((self.sequence_length, num_nodes, num_nodes, num_edge_features), dtype=np.float32)

        for t, edge_attr_dict in enumerate(self.edge_attr_seq):
            for (i, j), edge_attr in edge_attr_dict.items():
                edge_attr_array[t, i, j, :] = np.hstack([edge_attr['relative_position'],
                                                         edge_attr['relative_velocity'],
                                                         edge_attr['relative_acceleration']])

        
        node_features_tensor = torch.tensor(node_features_numpy, dtype=torch.float32).unsqueeze(0)  
        adj_matrix_tensor = torch.tensor(adj_matrix_numpy, dtype=torch.float32).unsqueeze(0)  
        edge_attr_tensor = torch.tensor(edge_attr_array, dtype=torch.float32).unsqueeze(0)  

        # Step 4: Get model prediction
        with torch.no_grad():
            model_prediction = self.model(node_features_tensor, adj_matrix_tensor, edge_attr_tensor)
        predicted_trajectory = model_prediction.cpu().numpy()

        # Step 5: Extract model's predicted (x, y)
        predicted_x = predicted_trajectory[0, -1, 0]  
        predicted_y = predicted_trajectory[0, -1, 1]  
        predicted_yaw = math.radians(predicted_trajectory[0, -1, 2])  

        # Step 6: Refine prediction using kinematic model
        transform = ego_vehicle.get_transform()
        velocity = ego_vehicle.get_velocity()
        vx, vy = velocity.x, velocity.y
        speed = math.sqrt(vx**2 + vy**2)
        control = ego_vehicle.get_control()
        steering_angle = math.radians(control.steer * 30)

        
        refined_x = transform.location.x + speed * math.cos(predicted_yaw) * dt
        refined_y = transform.location.y + speed * math.sin(predicted_yaw) * dt
        refined_yaw = predicted_yaw + (speed / wheelbase) * math.tan(steering_angle) * dt

        # # Step 6: Weighted combination of model and kinematic outputs
        model_weight =   0.7
        dynamics_weight =   0.3

        final_x = model_weight * predicted_x + dynamics_weight * refined_x
        final_y = model_weight * predicted_y + dynamics_weight * refined_y
        final_yaw = model_weight * predicted_yaw + dynamics_weight * refined_yaw

        # Step 7: Smooth the predictions
        if self.previous_predicted_x is not None and self.previous_predicted_y is not None:
            final_x = 0.7 * final_x + 0.3 * self.previous_predicted_x
            final_y = 0.7 * final_y + 0.3 * self.previous_predicted_y

        self.previous_predicted_x = final_x
        self.previous_predicted_y = final_y
        self.predicted_positions.append((final_x, final_y))
        self.actual_positions.append((ego_vehicle_features[0], ego_vehicle_features[1]))

    

        self.prediction_step += 1

        
        if self.prediction_step >= 500:
            self.plot_trajectory()
            self.save_predictions("predictions1.csv")
            raise StopIteration("Simulation stopped at step 180")

    
    def plot_trajectory(self):
        """Plot the trajectory with updated styles."""
        steps = np.arange(1, self.prediction_step + 1)
        predicted_xs, predicted_ys = zip(*self.predicted_positions)
        actual_xs, actual_ys = zip(*self.actual_positions)
       
        plt.figure(figsize=(20, 10))

        # Subplot for X positions over steps
        plt.subplot(1, 2, 1)
        plt.plot(steps, actual_xs, label="Actual Position X", color='blue', linestyle='-', marker='o', alpha=0.7, linewidth=0.8)
        plt.plot(steps, predicted_xs, label="Predicted Position X", color='red', linestyle='--', marker='x', alpha=0.6, linewidth=0.8)
        plt.xlabel("Timestep")
        plt.ylabel("Position X (meter)")
        plt.legend(fontsize=26, loc='lower right')
        plt.tight_layout()
        plt.grid(False)  

        # Subplot for Y positions over steps
        plt.subplot(1, 2, 2)
        plt.plot(steps, actual_ys, label="Actual Position Y", color='blue', linestyle='-', marker='o', alpha=0.7)
        plt.plot(steps, predicted_ys, label="Predicted Position Y", color='red', linestyle='--', marker='x', alpha=0.6)
        plt.xlabel("Timestep")
        plt.ylabel("Position Y (meter)")
        plt.legend(fontsize=26, loc='lower right')
        plt.tight_layout()
        plt.grid(False) 

        # Show both subplots
        plt.tight_layout()
        plt.savefig('output_image.pdf', format='pdf')
        plt.show()
        self.error_analysis(predicted_xs, predicted_ys, actual_xs, actual_ys)
    def error_analysis(self, predicted_xs, predicted_ys, actual_xs, actual_ys):
        """Perform error analysis and calculate error metrics."""
        # Convert to numpy arrays for easier computation
        predicted_xs = np.array(predicted_xs)
        predicted_ys = np.array(predicted_ys)
        actual_xs = np.array(actual_xs)
        actual_ys = np.array(actual_ys)

        # Calculate Mean Squared Error (MSE)
        mse_x = mean_squared_error(actual_xs, predicted_xs)
        mse_y = mean_squared_error(actual_ys, predicted_ys)
        rmse_x = np.sqrt(mse_x)
        rmse_y = np.sqrt(mse_y)

        # Calculate Mean Absolute Error (MAE)
        mae_x = mean_absolute_error(actual_xs, predicted_xs)
        mae_y = mean_absolute_error(actual_ys, predicted_ys)

        # Calculate final position error
        final_error_x = np.abs(actual_xs[-1] - predicted_xs[-1])
        final_error_y = np.abs(actual_ys[-1] - predicted_ys[-1])

        # Print the results
        print(f"Mean Squared Error (X): {mse_x}")
        print(f"Mean Squared Error (Y): {mse_y}")
        print(f"Root Mean Squared Error (X): {rmse_x}")
        print(f"Root Mean Squared Error (Y): {rmse_y}")
        print(f"Mean Absolute Error (X): {mae_x}")
        print(f"Mean Absolute Error (Y): {mae_y}")
        print(f"Final Position Error (X): {final_error_x}")
        print(f"Final Position Error (Y): {final_error_y}")

        # Plot the error over steps
        self.plot_error_over_steps(predicted_xs, predicted_ys, actual_xs, actual_ys)

    def plot_error_over_steps(self, predicted_xs, predicted_ys, actual_xs, actual_ys):
        """Plot the error (difference between actual and predicted positions) over steps."""
        steps = np.arange(1, self.prediction_step + 1)
        error_x = np.abs(actual_xs - predicted_xs)
        error_y = np.abs(actual_ys - predicted_ys)

        plt.figure(figsize=(10, 5), dpi=150)

        # Subplot for X position error over steps
        plt.subplot(1, 2, 1)
        plt.plot(steps, error_x, color='green')
        plt.xlabel("Timestep")
        plt.ylabel("Error of Position X (meter)")
        plt.tight_layout()
        plt.grid(False)

        # Subplot for Y position error over steps
        plt.subplot(1, 2, 2)
        plt.plot(steps, error_y, color='orange')
        plt.xlabel("Timestep")
        plt.ylabel("Error of Position Y (meter)")
        plt.grid(False)

        # Show both subplots
        plt.tight_layout()
        plt.savefig('output_image1.pdf', format='pdf')
        plt.show()
# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)


        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        clock = pygame.time.Clock()
 
        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            step_count += 1
            gnss_sensor = world.gnss_sensor
            gnss_csv_writer.writerow([step_count, gnss_sensor.lat, gnss_sensor.lon])
            control = world.player.get_control()

            steering_angles.append(control.steer)
            throttle.append(control.throttle)

            
           
            other_vehicles = [actor for actor in world.world.get_actors().filter('vehicle.*') if actor.id != world.player.id]
            predictor.predict_next_action(world.player, other_vehicles)

           
            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)
            csv_writer.writerow([predictor.prediction_step, throttle[-1]])
        
        predictor.plot_trajectory()

    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='Restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="Select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal)',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--num_variations',
        help='Number of random seed variations (default: 30)',
        default=30,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        
        for scenario_variation in range(args.num_variations):
            seed = args.seed if args.seed is not None else random.randint(0, 10000)
            print(f"\nRunning scenario variation {scenario_variation + 1}/{args.num_variations} with seed {seed}")
            random.seed(seed)  

            
            game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')



if __name__ == '__main__':
    main()
