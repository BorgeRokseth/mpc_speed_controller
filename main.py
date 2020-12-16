
from typing import NamedTuple, List
from .model import ShipConfiguration, EnvironmentConfiguration, \
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


def make_mpc_simulation_model(initial_conditions: InitialConditions,
                              simulation_time: float,
                              delta_north: float,
                              delta_east: float):
    ''' Set up simulation parameters for the simulated ship.
    '''
    start_way_point = WayPoint(north=0, east=0)
    way_point = WayPoint(north=1200 + delta_north, east=1000 + delta_east)
    end_way_point = WayPoint(north=2000, east=2000)
    route = [start_way_point,
             way_point,
             end_way_point]

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
        integration_step=1,
        simulation_time=simulation_time
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

def run_mpc_simulation(ship: ShipModelSimplifiedPropulsion, speed_setpoint_m_per_s: float):
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

    return pd.DataFrame().from_dict(ship.simulation_results)


def calculate_cost(x: List[float]):
    '''

    :param x: List[north coord of waypoint, east coord of waypoint]
    :return: Fuel cost
    '''
    dn_wp = x[0]
    de_wp = x[1]

    x0 = InitialConditions(
        north_m=0,
        east_m=0,
        yaw_rad=45 * np.pi / 180,
        u_m_per_s=0,
        v_m_per_s=0,
        r_rad_per_s=0,
        thrust_n=0
    )

    ship = make_mpc_simulation_model(initial_conditions=x0,
                                     simulation_time=500,
                                     delta_north=dn_wp,
                                     delta_east=de_wp)

    simulation_results = run_mpc_simulation(ship=ship, speed_setpoint_m_per_s=7)

    cost = simulation_results['fuel consumption me [kg]'].iloc[-1] ** 2

    print(cost)
    return cost


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

def optimize():
    t0 = time()

    upper_boundary = [800, 800]
    lower_boundary = [-800, -800]

    optimal_input, optimal_cost = pso(func=calculate_cost,
                                      lb=lower_boundary,
                                      ub=upper_boundary,
                                      minfunc=1e-6,
                                      swarmsize=70)
    t1 = time()
    print(optimal_cost)
    print(optimal_input)
    print(t1 - t0)
    return optimal_input

def test_optimization_results(u: List[float]):

    x0 = InitialConditions(
        north_m=0,
        east_m=0,
        yaw_rad=45 * np.pi / 180,
        u_m_per_s=0,
        v_m_per_s=0,
        r_rad_per_s=0,
        thrust_n=0
    )

    ship = make_mpc_simulation_model(initial_conditions=x0,
                                     simulation_time=10000,
                                     delta_north=u[0],
                                     delta_east=u[1])

    simulation_results = run_mpc_simulation(ship=ship, speed_setpoint_m_per_s=7)

    map_fig, map_ax = plt.subplots()
    plot_map_ax(simulation_data=simulation_results, simulated_ship=ship, ax=map_ax)

    plt.show()


if __name__ == "__main__":

    u = optimize()
    test_optimization_results(u)

