class Token:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.sequence = [Depot]  # Starts at depot
        self.cost_value = 0
        self.is_closed = False

    def add_node(self, customer, edge_cost):
        # Check if adding this node exceeds max trip cost
        if self.cost_value + edge_cost <= MAX_TRIP_COST:
            self.sequence.append(customer)
            self.cost_value += edge_cost
            return True
        else:
            # Cannot extend. Signal that a new token is needed.
            return False

class VRP_Solution:
    def __init__(self):
        self.vehicles = {} # Map vehicle_id -> List[Token]

    def assign_customer(self, vehicle_id, customer, edge_cost):
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = []

        vehicle_tokens = self.vehicles[vehicle_id]
        
        # Try to extend the last token
        if vehicle_tokens:
            last_token = vehicle_tokens[-1]
            if not last_token.is_closed and last_token.add_node(customer, edge_cost):
                return # Successfully added to existing token

        # If extension failed or no tokens exist, create a NEW token (New Route Hit)
        new_token = Token(vehicle_id)
        new_token.add_node(customer, edge_cost) # Add customer to new trip
        # Note: In a real model, you must ensure the new trip starts from the 
        # vehicle's current location (often Depot) or handle the return trip cost.
        new_token.sequence.append(Depot) # Close the trip
        new_token.cost_value += edge_cost # Add return cost
        
        vehicle_tokens.append(new_token)