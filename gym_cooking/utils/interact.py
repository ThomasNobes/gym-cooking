from utils.core import *
import numpy as np

def interact(agent, world):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    gs = world.get_gridsquare_at((action_x, action_y))

    # if floor in front --> move to that square
    if isinstance(gs, Floor): #and gs.holding is None:
        agent.move_to(gs.location)

    # if holding something
    elif agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            obj = agent.holding
            gs.acquire(obj)
            agent.release()
            if obj.is_deliverable():
                gs.release()  # remove delivery from kitchen
                for recipe in world.active_orders:
                    if str(obj) == recipe.get_ingredients():
                        world.active_orders.remove(recipe)
                        break
                print('\nDelivered {}!'.format(obj.full_name))
        
        # if food spawner in front --> do not interact
        elif isinstance(gs, FoodSpawner):
            pass

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects = False)

            if mergeable(agent.holding, obj):
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                # if playable version, merge onto counter first
                if world.arglist.play:
                    gs.acquire(agent.holding)
                    agent.release()


        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play:
                # normally chop, but if in playable game mode then put down first
                obj.chop()
            elif isinstance(gs, Trash):
                agent.release()
                world.remove(obj) # remove obj from world
                if obj.contains('Plate'):
                    new_plate = Object(
                                location=obj.location,
                                contents=RepToClass['p']())
                    agent.acquire(new_plate)
                    world.insert(new_plate)
            else:
                gs.acquire(obj) # obj is put onto gridsquare
                agent.release()
                assert world.get_object_at(gs.location, obj, find_held_objects =\
                    False).is_held == False, "Verifying put down works"

    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            # if in playable game mode, then chop raw items on cutting board
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play:
                obj.chop()
            else:
                held_obj = gs.release()
                assert held_obj == obj, "Verifying held object is the same as object on gridsquare"
                agent.acquire(held_obj)
                if isinstance(gs, FoodSpawner): # add new food to world if spawner
                    world.insert(held_obj)

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass
