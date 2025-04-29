import warnings

from abc import ABC

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

warnings.filterwarnings("ignore", message=".*DeprecationWarning.*")


class Problem:
    def __init__(
        self,
        travel_time_mtx,
        earliest_time,
        latest_time,
        service_time,
        setup_time,
        demand,
        capacity,
        max_tasks,
        n_pickup,
        n_dropoff,
        coords,
        fixed,
        secured,
        n,
        n_vehicles,
    ):
        self.travel_time_mtx = travel_time_mtx
        self.earliest_time = earliest_time
        self.latest_time = latest_time
        self.service_time = service_time
        self.setup_time = setup_time
        self.demand = demand
        self.capacity = capacity
        self.max_tasks = max_tasks
        self.n_pickup = n_pickup
        self.n_dropoff = n_dropoff
        self.coords = coords
        self.fixed = fixed
        self.secured = secured
        self.n = n
        self.n_vehicles = n_vehicles


class BaseSolver(ABC):
    def __init__(self):
        pass

    def update_problem(self, problem: Problem):
        self.problem = problem

    def solve(self, problem: Problem):
        raise NotImplementedError


class OrtoolsSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def update_problem(self, problem: Problem):
        super().update_problem(problem)

        # Problem settings
        n_pickup = problem.n_pickup
        n_dropoff = problem.n_dropoff
        problem.n - n_pickup - n_dropoff
        self.n_isolated_dropoff = n_dropoff - n_pickup

        # Paired pickup and dropoff nodes
        self.pickups_deliveries = (
            np.array([i for i in range(2 * n_pickup)]).reshape(2, n_pickup).T
        )

        # Preparing input data
        self.input_data = {
            "distance_matrix": [],
            "num_vehicles": problem.n_vehicles,
            "starts": [
                i for i in range(2 * n_pickup + self.n_isolated_dropoff, problem.n)
            ],
            "ends": [problem.n] * problem.n_vehicles,
            "pickups_deliveries": self.pickups_deliveries,
            "isolated_dropoff": [
                i for i in range(2 * n_pickup, 2 * n_pickup + self.n_isolated_dropoff)
            ],
            "demands": problem.demand,
            "max_tasks": problem.max_tasks,
            "vehicle_capacities": problem.capacity,
            "earliest_time": problem.earliest_time,
            "latest_time": problem.latest_time,
            "service_time": problem.service_time,
            "setup_time": problem.setup_time,
            "fixed": problem.fixed,
            "secured": problem.secured,
        }

        mtx_with_dummy_node = np.zeros((problem.n + 1, problem.n + 1), dtype=int)
        mtx_with_dummy_node[:-1, :-1] = problem.travel_time_mtx
        self.input_data["distance_matrix"] = mtx_with_dummy_node.astype(int).tolist()

    def solve(
        self,
        problem: Problem,
        vehicle_max_travel,
        global_span_cost_coefficient=100000,
        use_soft_time_windows=True,
        timelimit=10,
        lateness_penalty=100,
    ):
        self.update_problem(problem)

        # Create routing manager and model
        manager = pywrapcp.RoutingIndexManager(
            len(self.input_data["distance_matrix"]),
            self.input_data["num_vehicles"],
            self.input_data["starts"],
            self.input_data["ends"],
        )
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.input_data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance dimension
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            int(self.input_data["setup_time"][0] + self.input_data["service_time"][0]),
            int(vehicle_max_travel),
            False,
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        # distance_dimension.SetGlobalSpanCostCoefficient(global_span_cost_coefficient)

        # Add time window constraints for non-depot locations
        depot_start_index = self.problem.n_pickup + self.problem.n_dropoff
        for location_idx, (
            earliest_time,
            latest_time,
            service_time,
            setup_time,
        ) in enumerate(
            zip(
                self.input_data["earliest_time"][:depot_start_index],
                self.input_data["latest_time"][:depot_start_index],
                self.input_data["service_time"][:depot_start_index],
                self.input_data["setup_time"][:depot_start_index],
            )
        ):
            index = manager.NodeToIndex(location_idx)

            distance_dimension.SetCumulVarSoftLowerBound(index, int(earliest_time), 0)
            distance_dimension.SetCumulVarSoftUpperBound(
                index, int(earliest_time), lateness_penalty
            )

        # Add time window constraints for vehicle start nodes
        for vehicle_id in range(self.input_data["num_vehicles"]):
            route_start_index = routing.Start(vehicle_id)
            routing.End(vehicle_id)

            earliest_time = self.input_data["earliest_time"][
                vehicle_id + depot_start_index
            ]
            self.input_data["latest_time"][vehicle_id + depot_start_index]

            distance_dimension.SetCumulVarSoftLowerBound(
                route_start_index, int(earliest_time), 0
            )
            distance_dimension.SetCumulVarSoftUpperBound(
                route_start_index, int(earliest_time), lateness_penalty
            )

        # Demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.input_data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            self.input_data["vehicle_capacities"],
            True,
            "Capacity",
        )

        # Add paired pickup and delivery constraints
        for request in self.input_data["pickups_deliveries"]:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index)
                <= distance_dimension.CumulVar(delivery_index)
            )

        # Set first solution strategy and metaheuristics
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = timelimit

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        # Extract routes
        if solution:
            return self._get_routes_from_solution(solution, routing, manager)
        else:
            raise Exception(self._get_routing_status(routing))

    def _get_routes_from_solution(self, solution, routing, manager):
        routes = []
        route_length = []
        for route_id in range(routing.vehicles()):
            index = routing.Start(route_id)
            route = [manager.IndexToNode(index)]
            route_distance = 0
            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, route_id
                )
                route.append(manager.IndexToNode(index))
            routes.append(route[:-1])  # Remove the dummy node
            route_length.append(route_distance)
        return routes, sum(route_length)

    def _get_routing_status(self, routing):
        status_codes = {
            0: "ROUTING_NOT_SOLVED: Problem not solved yet.",
            1: "ROUTING_SUCCESS: Problem solved successfully.",
            2: "ROUTING_FAIL: No solution found to the problem.",
            3: "ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.",
            4: "ROUTING_INVALID: Model, model parameters, or flags are not valid.",
        }
        return status_codes.get(routing.status(), "ROUTING_UNKNOWN_STATUS")
