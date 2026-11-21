"""Network feature names (Plan §7.8), kept free of heavy imports so serving code can use them
without installing networkx. `finsight.graph.network_features` computes them."""

POINT_IN_TIME_FEATURES = [
    "sender_tx_10m", "sender_tx_1h", "sender_tx_24h", "sender_distinct_receivers_24h",
    "receiver_tx_in_24h", "receiver_distinct_senders_24h", "receiver_distinct_senders_7d",
    "sec_since_sender_last_inflow", "forward_ratio", "is_pass_through", "chain_depth",
    "device_distinct_accounts", "ip_distinct_accounts",
]
GLOBAL_FEATURES = [
    "sender_pagerank", "receiver_pagerank", "sender_community_size", "receiver_community_size",
    "same_community", "receiver_clustering",
]
NETWORK_FEATURES = POINT_IN_TIME_FEATURES + GLOBAL_FEATURES
