"""
simulator.py

Wireless network simulation with:
- Two access points (APs): Main AP and Extender AP.
- Five simulated clients.
- Two association policies:
    1: Baseline RSSI-based policy.
    2: Machine-learning-based policy (uses a trained classifier).

The simulator:
- Assigns each client to an AP according to a given policy.
- Computes per-client throughput given AP load and backhaul limits.
- Counts "sticky" clients
"""

import numpy as np




class AP:
    """
    Represents a Wi-Fi access point in the simulation.

    AP:
        name (str): Identifier for the AP ("Main" or "Ext").
        backhaul_capacity (float or None): Maximum Mbps allowed by the backhaul
            - For the Main AP, this is typically None (assumed unconstrained).
            - For the Extender AP, this models a wireless backhaul link.
        connected (list[Client]): List of Client objects currently associated.
    """

    def __init__(self, name, backhaul_capacity=None):
        self.name = name
        self.backhaul_capacity = backhaul_capacity
        self.connected = []

    def load(self) -> int:
        """
        Returns the current AP load, defined as the number of connected clients
        """
        return len(self.connected)


class Client:
    """
    Represents a simulated Wi-Fi client

    Client:
        name (str): Client identifier (e.g., "SimClient-1").
        rssi_main (int): RSSI value towards the Main AP (in dBm, negative).
        rssi_ext (int): RSSI value towards the Ext AP (in dBm, negative).
        associated_ap (str or None): Name of the AP this client is connected to.
    """

    def __init__(self, name, rssi_main, rssi_ext):
        self.name = name
        self.rssi_main = rssi_main
        self.rssi_ext = rssi_ext
        self.associated_ap = None



def compute_throughput(ap: AP) -> float:
    """
    Computes per-client throughput for an AP based on:
    - A base radio capacity (simplified constant: 300 Mbps).
    - Optional backhaul capacity for the Ext AP.
    - The number of connected clients

    Returns:
        float: Throughput (Mbps) each connected client receives on this AP.
               Returns 0 if the AP has no clients.
    """
    # Base Wi-Fi radio capacity
    base_capacity = 300.0  # Mbps

    # If this AP has a backhaul capacity, it becomes the bottleneck
    if ap.backhaul_capacity is not None:
        base_capacity = min(base_capacity, ap.backhaul_capacity)

    # No clients means zero throughput
    if len(ap.connected) == 0:
        return 0.0

    # Simplified model: equal share of capacity among all clients
    return base_capacity / len(ap.connected)




def baseline_policy(client: Client, main_ap: AP, ext_ap: AP) -> AP:
    """
    Baseline RSSI-based association policy.

    Logic:
        - If the client sees stronger RSSI from the Ext AP, join Ext.
        - Otherwise, join Main.

    This approximates the behavior of typical extenders which only
    consider signal strength, not load or backhaul quality.

    Returns:
        AP: The chosen AP for this client.
    """
    if client.rssi_ext > client.rssi_main:
        return ext_ap
    else:
        return main_ap


def ml_policy(client: Client, main_ap: AP, ext_ap: AP, clf) -> AP:
    """
    Machine-learning-based association policy.

    Uses a trained classifier (clf) to predict the best AP given:
        - RSSI to Main AP
        - RSSI to Ext AP
        - Ext AP backhaul capacity
        - Current load (number of clients) on Main AP
        - Current load on Ext AP

    The classifier outputs:
        0 - Main AP is better
        1 - Ext AP is better

    Args:
        client (Client): The client to associate
        main_ap (AP): Main AP object
        ext_ap (AP): Extender AP object
        clf: Trained scikit-learn classifier

    Returns:
        AP: The AP predicted to give better throughput.
    """
    # Build feature vector in the same order as training
    features = np.array([
        client.rssi_main,
        client.rssi_ext,
        ext_ap.backhaul_capacity,
        main_ap.load(),
        ext_ap.load()
    ]).reshape(1, -1)

    prediction = clf.predict(features)[0]  # 0 or 1
    return main_ap if prediction == 0 else ext_ap


def simulate_run(policy_fn, clf=None):
    """
    Runs a single simulation run for all five clients using
    the given association policy.

    Steps:
        1. Create Main AP and Ext AP
            - Main AP has no explicit backhaul limit
            - Ext AP gets a random backhaul capacity between 120–180 Mbps
        2. Create 5 clients with fixed RSSI values
        3. For each client:
            a) Use the chosen policy to pick an AP
            b) Compute which AP would be ideal if we only cared about
               throughput in this moment
            c) If chosen AP != ideal AP - count as a sticky event.
            d) Attach client to chosen AP
        4. After all clients are associated, compute per-client throughput
           based on AP load and backhaul
        5. Build a result list: [client_name, associated_ap_name, throughput]

    Args:
        policy_fn: The association policy function to use
            - baseline_policy(client, main_ap, ext_ap)
            - ml_policy(client, main_ap, ext_ap, clf)
        clf: Trained ML model

    Returns:
        results (list[list]) Each inner list is:
            [client_name, ap_name, throughput]
        sticky_events (int): Number of times the chosen AP was not
            the instantaneous throughput-optimal AP
    """
    # Create APs: Main , Ext 
    main_ap = AP("Main")
    ext_ap = AP("Ext", backhaul_capacity=np.random.randint(120, 180))

    # Fixed RSSI per client to keep scenarios somewhat stable
    clients = [
        Client("SimClient-1", -55, -60),
        Client("SimClient-2", -65, -50),
        Client("SimClient-3", -70, -57),
        Client("SimClient-4", -52, -69),
        Client("SimClient-5", -67, -48),
    ]

    sticky_events = 0
    results = []

    # Decide AP for each client
    for client in clients:
        # Use baseline or ML policy depending on whether clf is provided
        if clf is None:
            chosen_ap = policy_fn(client, main_ap, ext_ap)
        else:
            chosen_ap = policy_fn(client, main_ap, ext_ap, clf)

        # Compute ideal AP based purely on throughput at this moment
        ideal_thr_main = 300 / (main_ap.load() + 1)
        ideal_thr_ext = min(300, ext_ap.backhaul_capacity) / (ext_ap.load() + 1)
        ideal_ap = main_ap if ideal_thr_main > ideal_thr_ext else ext_ap

        # If chosen AP is not the ideal throughput AP, count a sticky event
        if chosen_ap != ideal_ap:
            sticky_events += 1

        # Attach the client to the chosen AP and record association
        chosen_ap.connected.append(client)
        client.associated_ap = chosen_ap.name

    # After all clients are connected, compute the throughput they get
    thr_main = compute_throughput(main_ap)
    thr_ext = compute_throughput(ext_ap)

    # Build results per client: which AP, and what throughput they see
    for client in clients:
        if client.associated_ap == "Main":
            thr = thr_main
        else:
            thr = thr_ext

        results.append([client.name, client.associated_ap, thr])

    return results, sticky_events
