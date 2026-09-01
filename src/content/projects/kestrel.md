# Kestrel: Distributed Key-Value Store

```
  _  __          _             _ 
 | |/ /___  ___ | |_ _ __ ___ | |
 | ' // _ \/ __|| __| '__/ _ \| |
 | . \  __/\__ \| |_| | |  __/| |
 |_|\_\___||___/ \__|_|  \___||_|
```

> **Distributed, fault-tolerant key-value store with RESP2 protocol compatibility, Raft consensus, AOF durability, and consistent hash sharding built from scratch in Go.**

---

## 🎯 Overview & Problem Statement

Modern distributed applications demand reliable, low-latency in-memory data stores. However, existing monolithic solutions often present complex configuration overhead or opaque failure modes.

**Kestrel** is a distributed key-value storage engine engineered from the ground up in **Go**. It implements the **RESP2 protocol**, allowing drop-in compatibility with native `redis-cli` clients. Kestrel guarantees zero data loss via configurable Append-Only File (AOF) durability, automatic leader election via **Raft consensus**, horizontal scaling with consistent hashing, and live cluster observability through a **Bubbletea TUI** and Prometheus.

---

## 🏗️ System Architecture & Data Flow

```
                      +-----------------------------+
                      |   redis-cli / TCP Clients   |
                      +--------------+--------------+
                                     |
                         RESP2 TCP Traffic (:6379)
                                     v
                      +-----------------------------+
                      |   Consistent Hash Proxy     |
                      |   (Horizontal Sharding)     |
                      +-------+-------------+-------+
                              |             |
                     Node A   |             | Node B / C
                              v             v
            +-------------------------------------------------+
            |                 Kestrel Node                    |
            |                                                 |
            |   +-----------------------------------------+   |
            |   |          RESP2 Protocol Parser          |   |
            |   +--------------------+--------------------+   |
            |                        |                        |
            |                        v                        |
            |   +--------------------+--------------------+   |
            |   |       In-Memory State Store (Map)       |   |
            |   |  (GET, SET, DEL, EXPIRE, HSET, LPUSH)   |   |
            |   +----------+-------------------+----------+   |
            |              |                   |              |
            |              v                   v              |
            |   +--------------------+   +----------------+   |
            |   |  AOF Durability    |   | Raft Consensus |   |
            |   |  (always/everysec) |   | (Leader Elec & |   |
            |   |  (WAL / BoltDB)    |   |  Replication)  |   |
            |   +--------------------+   +----------------+   |
            +-------------------------------------------------+
                                     |
                       Metrics & Cluster State Stream
                                     v
                      +-----------------------------+
                      |   Bubbletea TUI / Grafana   |
                      +-----------------------------+
```

---

## ⚡ Key Features & Engineering Innovations

- **Custom RESP2 Protocol Engine**: Complete TCP server implementing the Redis Serialization Protocol (RESP2) supporting `GET`, `SET`, `DEL`, `EXPIRE`, `TTL`, `HSET`, `LPUSH`, `ZADD`, `PING`, and `INFO` commands.
- **AOF Durability & Crash Recovery**: Configurable fsync policies (`always`, `everysec`, `no`) paired with background snapshotting. Replays **367,000 writes in ~318ms** during crash recovery with zero data loss.
- **Distributed Consensus (Raft / VSR)**: Leader election and synchronized log replication across 3-node clusters, achieving leader failover in **~1.5s** (Raft) and **~1.1s** (Viewstamped Replication).
- **Consistent Hash Sharding Proxy**: Stateless sharding proxy with virtual nodes delivering a balanced (~50/50) key distribution across independent storage nodes.
- **Terminal UI & Observability**: Interactive **Bubbletea TUI** dashboard displaying live memory usage, cluster topology, replication lag, and Prometheus metrics export.

---

## 📊 Benchmark Metrics & Performance

| Metric | Measured Value | Comparison / Notes |
| :--- | :--- | :--- |
| **Single-Node Throughput (YCSB-lite)** | **~12,904 ops/sec** | ~11.63ms p99 latency (unconstrained up to 246k ops/sec) |
| **Redis Baseline Throughput** | ~34,427 ops/sec | Reference redis-server baseline |
| **3-Node Cluster Throughput** | **~11,250 ops/sec** | 1–5ms replication lag across nodes |
| **AOF Recovery Replay Time** | **~318ms** | Full replay of 367,000 write operations |
| **Raft Leader Failover Time** | **~1.5s** | Automatic detection & new leader elected |
| **VSR Leader Failover Time** | **~1.1s** | Viewstamped Replication phase failover |

---

## 🛠️ Tech Stack & Badges

- **Language**: `Go (Golang 1.22+)`
- **Protocol**: `RESP2 (Redis Serialization Protocol)`
- **Consensus & Clustering**: `Raft Consensus Algorithm`, `Viewstamped Replication (VSR)`, `Consistent Hashing`
- **Storage & Durability**: `BoltDB`, `Write-Ahead Logging (WAL)`, `Append-Only File (AOF)`
- **Observability & UI**: `Bubbletea TUI (Charm.sh)`, `Prometheus`, `Grafana`
- **Testing & Benchmarking**: `Custom Go YCSB-lite Load Harness`, `Go Test Suite`

---

## 🔗 Repository & Deployment

- **GitHub Repository**: [https://github.com/ladsad/kestrel](https://github.com/ladsad/kestrel)
- **Local Run**: `go run cmd/kestrel/main.go --port=6379 --aof=everysec`
