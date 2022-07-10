"""
https://www.kaggle.com/code/nadare/word-tour-experiment/notebook easy to use?
https://github.com/joisino/wordtour original

#pip install ortools

"""

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import os

import pandas as pd
import numpy as np

data = {}
embed= np.load("")#N,k
cos = torch.nn.CosineSimilarity(dim=1)
data["distance_matrix"] = cos(embed, embed).astype(np.int64).tolist()
data["num_vehicles"] = 1
data["depot"] = 0

def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes[0]

manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

solution = routing.SolveWithParameters(search_parameters)
routes = get_routes(solution, routing, manager)[:-1]#N*1#埋め込みの質が高くないと効果発揮しない


