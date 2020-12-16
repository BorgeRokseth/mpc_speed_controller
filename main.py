
from typing import NamedTuple, List
from model import ShipConfiguration, EnvironmentConfiguration, \
    MachineryModeParams, MachineryMode, MachineryModes, \
    SimplifiedPropulsionMachinerySystemConfiguration, SimplifiedPropulsionSimulationConfiguration, \
    ShipModelSimplifiedPropulsion, WayPoint
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso
from time import time
import numpy as np


class InitialConditions(NamedTuple):
    north_m: float
    east_m: float
    yaw_rad: float
    u_m_per_s: float
    v_m_per_s: float
    r_rad_per_s: float
    thrust_n: float


def make_actual_ship_simulation_model(initial_conditions: InitialConditions,
                                      route: List[WayPoint]):
    main_engine_capacity = 2160e3
    diesel_gen_capacity = 510e3
    hybrid_shaft_gen_as_generator = 'GEN'
    hybrid_shaft_gen_as_motor = 'MOTOR'
    hybrid_shaft_gen_as_offline = 'OFF'

    ship_config = ShipConfiguration(
        coefficient_of_deadweight_to_displacement=0.7,
        bunkers=200000,
        ballast=200000,
        length_of_ship=80,
        width_of_ship=16,
        added_mass_coefficient_in_surge=0.4,
        added_mass_coefficient_in_sway=0.4,
        added_mass_coefficient_in_yaw=0.4,
        dead_weight_tonnage=3850000,
        mass_over_linear_friction_coefficient_in_surge=130,
        mass_over_linear_friction_coefficient_in_sway=18,
        mass_over_linear_friction_coefficient_in_yaw=90,
        nonlinear_friction_coefficient__in_surge=2400,
        nonlinear_friction_coefficient__in_sway=4000,
        nonlinear_friction_coefficient__in_yaw=400
    )
    env_config = EnvironmentConfiguration(
        current_velocity_component_from_north=0,
        current_velocity_component_from_east=0,
        wind_speed=0,
        wind_direction=0
    )

    pto_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=0,
        shaft_generator_state=hybrid_shaft_gen_as_generator
    )
    pto_mode = MachineryMode(params=pto_mode_params)

    mec_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=diesel_gen_capacity,
        shaft_generator_state=hybrid_shaft_gen_as_offline
    )
    mec_mode = MachineryMode(params=mec_mode_params)

    pti_mode_params = MachineryModeParams(
        main_engine_capacity=0,
        electrical_capacity=2 * diesel_gen_capacity,
        shaft_generator_state=hybrid_shaft_gen_as_motor
    )
    pti_mode = MachineryMode(params=pti_mode_params)

    mso_modes = MachineryModes(
        [pto_mode,
         mec_mode,
         pti_mode]
    )

    machinery_config = SimplifiedPropulsionMachinerySystemConfiguration(
        hotel_load=200e3,
        machinery_modes=mso_modes,
        max_rudder_angle_degrees=30,
        rudder_angle_to_yaw_force_coefficient=500e3,
        rudder_angle_to_sway_force_coefficient=50e3,
        thrust_force_dynamic_time_constant=30
    )

    simulation_setup = SimplifiedPropulsionSimulationConfiguration(
        route=route,
        initial_north_position_m=initial_conditions.north_m,
        initial_east_position_m=initial_conditions.east_m,
        initial_yaw_angle_rad=initial_conditions.yaw_rad,
        initial_forward_speed_m_per_s=initial_conditions.u_m_per_s,
        initial_sideways_speed_m_per_s=initial_conditions.v_m_per_s,
        initial_yaw_rate_rad_per_s=initial_conditions.r_rad_per_s,
        initial_thrust_force=initial_conditions.thrust_n,
        machinery_system_operating_mode=1,
        integration_step=0.8,
        simulation_time=8000
    )

    return ShipModelSimplifiedPropulsion(ship_config=ship_config,
                                         machinery_config=machinery_config,
                                         environment_config=env_config,
                                         simulation_config=simulation_setup)


def run_actual_ship_simulation(ship:ShipModelSimplifiedPropulsion,
                               speed_setpoint_m_per_s: float):
    i = 0
    draw_interval = 30
    time_since_last_ship_drawing = 0
    while ship.int.time < ship.int.sim_time and not ship.game_over:
        rudder_angle = -ship.rudderang_from_route()
        engine_load = ship.loadperc_from_speedref(speed_setpoint_m_per_s)

        ship.update_differentials(load_perc=engine_load, rudder_angle=rudder_angle)
        ship.integrate_differentials()

        ship.store_simulation_data(engine_load)

        # Make a drawing of the ship from above every 20 second
        if time_since_last_ship_drawing > draw_interval:
            ship.ship_snap_shot()
            time_since_last_ship_drawing = 0
        time_since_last_ship_drawing += ship.int.dt

        # Progress time variable and counter to the next time step
        ship.int.next_time()
        i += 1


class FuelConsumption:
    def __init__(self, initial_conditions: InitialConditions,
                 simulation_time: float,
                 route: List[WayPoint]):

        self.initial_conditions = initial_conditions
        self.simulation_time = simulation_time
        self.waypoints = route

        main_engine_capacity = 2160e3
        diesel_gen_capacity = 510e3
        hybrid_shaft_gen_as_generator = 'GEN'
        hybrid_shaft_gen_as_motor = 'MOTOR'
        hybrid_shaft_gen_as_offline = 'OFF'

        self.ship_config = ShipConfiguration(
            coefficient_of_deadweight_to_displacement=0.7,
            bunkers=200000,
            ballast=200000,
            length_of_ship=80,
            width_of_ship=16,
            added_mass_coefficient_in_surge=0.4,
            added_mass_coefficient_in_sway=0.4,
            added_mass_coefficient_in_yaw=0.4,
            dead_weight_tonnage=3850000,
            mass_over_linear_friction_coefficient_in_surge=130,
            mass_over_linear_friction_coefficient_in_sway=18,
            mass_over_linear_friction_coefficient_in_yaw=90,
            nonlinear_friction_coefficient__in_surge=2400,
            nonlinear_friction_coefficient__in_sway=4000,
            nonlinear_friction_coefficient__in_yaw=400
        )
        self.env_config = EnvironmentConfiguration(
            current_velocity_component_from_north=0,
            current_velocity_component_from_east=0,
            wind_speed=0,
            wind_direction=0
        )

        pto_mode_params = MachineryModeParams(
            main_engine_capacity=main_engine_capacity,
            electrical_capacity=0,
            shaft_generator_state=hybrid_shaft_gen_as_generator
        )
        pto_mode = MachineryMode(params=pto_mode_params)

        mec_mode_params = MachineryModeParams(
            main_engine_capacity=main_engine_capacity,
            electrical_capacity=diesel_gen_capacity,
            shaft_generator_state=hybrid_shaft_gen_as_offline
        )
        mec_mode = MachineryMode(params=mec_mode_params)

        pti_mode_params = MachineryModeParams(
            main_engine_capacity=0,
            electrical_capacity=2 * diesel_gen_capacity,
            shaft_generator_state=hybrid_shaft_gen_as_motor
        )
        pti_mode = MachineryMode(params=pti_mode_params)

        mso_modes = MachineryModes(
            [pto_mode,
             mec_mode,
             pti_mode]
        )

        self.machinery_config = SimplifiedPropulsionMachinerySystemConfiguration(
            hotel_load=200e3,
            machinery_modes=mso_modes,
            max_rudder_angle_degrees=30,
            rudder_angle_to_yaw_force_coefficient=500e3,
            rudder_angle_to_sway_force_coefficient=50e3,
            thrust_force_dynamic_time_constant=30
        )

    @staticmethod
    def run_simulation(ship: ShipModelSimplifiedPropulsion,
                       speed_setpoint_m_per_s: float):
        ''' Run simulations to generate output for cost calculation.
        '''
        i = 0
        draw_interval = 30
        time_since_last_ship_drawing = 0
        while ship.int.time < ship.int.sim_time and not ship.game_over:
            rudder_angle = -ship.rudderang_from_route()
            engine_load = ship.loadperc_from_speedref(speed_setpoint_m_per_s)

            ship.update_differentials(load_perc=engine_load, rudder_angle=rudder_angle)
            ship.integrate_differentials()

            ship.store_simulation_data(engine_load)

            # Make a drawing of the ship from above every 20 second
            if time_since_last_ship_drawing > draw_interval:
                ship.ship_snap_shot()
                time_since_last_ship_drawing = 0
            time_since_last_ship_drawing += ship.int.dt

            # Progress time variable and counter to the next time step
            ship.int.next_time()
            i += 1

        return ship.simulation_results['fuel consumption [kg]'][-1]

    def adjust_waypoints_according_to_input(self,
                                            north_adjustments: List[float],
                                            east_adjustments: List[float]):
        i = 0
        j = 0
        for way_point in self.waypoints:
            if i != 0 and i != len(self.waypoints) - 1:
                way_point.adjust_location(move_north=north_adjustments[j],
                                          move_east=east_adjustments[j])
                j += 1
            i += 1

    def cost_calculation(self, u):
        delta_north = u[0::2]
        delta_east = u[1::2]
        self.adjust_waypoints_according_to_input(north_adjustments=delta_north,
                                                 east_adjustments=delta_east)

        self.simulation_setup = SimplifiedPropulsionSimulationConfiguration(
            route=self.waypoints,
            initial_north_position_m=self.initial_conditions.north_m,
            initial_east_position_m=self.initial_conditions.east_m,
            initial_yaw_angle_rad=self.initial_conditions.yaw_rad,
            initial_forward_speed_m_per_s=self.initial_conditions.u_m_per_s,
            initial_sideways_speed_m_per_s=self.initial_conditions.v_m_per_s,
            initial_yaw_rate_rad_per_s=self.initial_conditions.r_rad_per_s,
            initial_thrust_force=self.initial_conditions.thrust_n,
            machinery_system_operating_mode=1,
            integration_step=1,
            simulation_time=self.simulation_time
        )
        ship = ShipModelSimplifiedPropulsion(ship_config=self.ship_config,
                                             machinery_config=self.machinery_config,
                                             environment_config=self.env_config,
                                             simulation_config=self.simulation_setup)
        fuel_usage = self.run_simulation(ship=ship, speed_setpoint_m_per_s=7)
        print('Fuel usage: ', self.run_simulation(ship=ship, speed_setpoint_m_per_s=7))
        return fuel_usage


class OptimizeWaypointPlacementInPredictionHorizon:
    def __init__(self,
                 x_0: InitialConditions,
                 waypoints: List[WayPoint]):
        self.fuel_consumption = FuelConsumption(initial_conditions=x_0,
                                                route=waypoints,
                                                simulation_time=5000)
        lower_north_boundary = -250
        lower_east_boundary = -250
        upper_north_boundary = 500
        upper_east_boundary = 500
        lower_north = []
        lower_east = []
        upper_north = []
        upper_east = []
        for i in range(0,len(waypoints)-2):
            lower_north.append(lower_north_boundary)
            lower_east.append(lower_east_boundary)
            upper_north.append(upper_north_boundary)
            upper_east.append(upper_east_boundary)
        self.lower_boundary = lower_north + lower_east
        self.upper_boundary = upper_north + upper_east

    def minimize_fuel_consumption(self):
        u_opt, u_cost = pso(func=self.fuel_consumption.cost_calculation,
                            lb=self.lower_boundary,
                            ub=self.upper_boundary,
                            minfunc=1e-6,
                            swarmsize=70)
        return u_opt, u_cost


def plot_map_ax(simulation_data: pd.DataFrame, simulated_ship: ShipModelSimplifiedPropulsion, ax):
    ax.plot(simulation_data['east position [m]'], simulation_data['north position [m]'])
    for x, y in zip(simulated_ship.ship_drawings[1], simulated_ship.ship_drawings[0]):
        ax.plot(x, y, color='black')

    for waypoint in simulated_ship.navigate.list_of_waypoints:
        ax.plot(waypoint.east, waypoint.north, 'bo')
        ax.add_patch(plt.Circle((waypoint.east, waypoint.north),
                                radius=simulated_ship.navigate.ra,
                                fill=False, color='grey'))
    ax.set_aspect('equal')


def deg2rad(deg: float):
    return deg * np.pi / 180


if __name__ == "__main__":

    route = [
        WayPoint(0, 0),
        WayPoint(1000, 1000),
        WayPoint(1000, 2000),
        WayPoint(2000, 3000),
        WayPoint(3000, 3500),
        WayPoint(4000, 4000),
    ]
    initial_states = InitialConditions(
        north_m=0,
        east_m=0,
        yaw_rad=deg2rad(45),
        u_m_per_s=0,
        v_m_per_s=0,
        r_rad_per_s=0,
        thrust_n=0
    )

    prediction_horizon_wp = [
        WayPoint(north=0, east=0),
        WayPoint(north=1000, east=750),
        WayPoint(north=2000, east=2000),
        WayPoint(north=2500, east=3000)
    ]

    opt = OptimizeWaypointPlacementInPredictionHorizon(x_0=initial_states,
                                                       waypoints=prediction_horizon_wp)
    optimal_setpoint = opt.minimize_fuel_consumption()
    #u = optimize()
    #test_optimization_results(u)

