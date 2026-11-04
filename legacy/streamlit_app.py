import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

if 'nodes' not in st.session_state:
    st.session_state.nodes = []
    st.session_state.pipes = []  # Critical missing line
    st.session_state.junction_count = 0
    st.session_state.tank_count = 0
    st.session_state.reservoir_count = 0
    st.session_state.pipe_count = 0

def generate_node_id(node_type):
    """Generate standardized node IDs"""
    if node_type == 'junction':
        st.session_state.junction_count += 1
        return f"J{st.session_state.junction_count}"
    elif node_type == 'tank':
        st.session_state.tank_count += 1
        return f"T{st.session_state.tank_count}"
    elif node_type == 'reservoir':
        st.session_state.reservoir_count += 1
        return f"R{st.session_state.reservoir_count}"
    return ""

def generate_pipe_id():
    """Generate standardized pipe IDs"""
    st.session_state.pipe_count += 1
    return f"P{st.session_state.pipe_count}"

def visualize_network():
    """Create network visualization"""
    G = nx.Graph()
    node_colors = {
        'junction': 'skyblue',
        'tank': 'gold',
        'reservoir': 'salmon'
    }

    # Add nodes
    for node in st.session_state.nodes:
        G.add_node(node['id'], type=node['type'])

    # Add edges
    for pipe in st.session_state.pipes:
        edge_attrs = {
            'label': pipe['id'],
            'style': 'dashed' if pipe['has_pump'] else 'solid'
        }
        if pipe['has_pump']:
            edge_attrs['label'] += f"\n{pipe['pump_power']}kW"
        G.add_edge(pipe['start'], pipe['end'], **edge_attrs)

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw nodes
    for node_type, color in node_colors.items():
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[n['id'] for n in st.session_state.nodes if n['type'] == node_type],
            node_color=color,
            node_size=1000,
            ax=ax
        )

    # Draw edges
    for edge in G.edges(data=True):
        style = edge[2].get('style', 'solid')
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(edge[0], edge[1])],
            style=style,
            width=2,
            ax=ax
        )
        edge_label = edge[2].get('label', '')
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(edge[0], edge[1]): edge_label},
            ax=ax
        )

    # Labels and legend
    nx.draw_networkx_labels(G, pos, ax=ax)
    legend_elements = [
        Circle(0, color='skyblue', label='Junction'),
        Circle(0, color='gold', label='Tank'),
        Circle(0, color='salmon', label='Reservoir'),
        plt.Line2D([0], [0], color='black', linestyle='-', label='Pipe'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Pump')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.title("Water Network Diagram")
    st.pyplot(fig)

def create_network_matrices(nodes, pipes):
    """Create matrices with proper K dimension alignment"""
    # Calculate components
    non_pumped = sum(1 for p in pipes if not p['has_pump'])
    pumped = len(pipes) - non_pumped
    junctions = sum(1 for n in nodes if n['type'] == 'junction')
    reservoirs = sum(1 for n in nodes if n['type'] == 'reservoir')
    tanks = sum(1 for n in nodes if n['type'] == 'tank')

    # Calculate matrix dimension using original formula
    K = (len(pipes) + 2 * non_pumped + junctions + reservoirs + tanks)

    # Initialize E matrix
    E = [[0] * K for _ in range(K)]

    #Initialize G matrix
    G = ['0'] * K

    # Set diagonals for non-pumped pipes
    for i in range(non_pumped):
        E[i][i] = 1
        G[i] = f'-C_aQ{i + 1}|Q{i + 1}|'

    # Set diagonals for tanks (bottom rows)
    tank_start = K - tanks
    for i in range(tank_start, K):
        E[i][i] = 1

    #Initialize F matrix
    F = ['0'] * K

    # Initialize A matrix with empty strings
    A = [['0'] * K for _ in range(K)]

    # Add coefficients for flows of pipes (2 per pipe)
    for i in range(non_pumped):
        # Pipe left coefficient (c1, c3, ...)
        A[i][non_pumped + pumped + i] = f'c{i + 1}'
        # # Pipe right coefficient (c2, c4, ...)
        A[i][(2*non_pumped) + pumped + i] = f'-c{i + 1}'

    # Add coefficients for pumped pipe flows
    for i in range(pumped):
        A[non_pumped + i][non_pumped + i] = f'1-s{non_pumped + i + 1}'
        # A[non_pumped + i][(3 * non_pumped) + pumped + i] = f's{non_pumped + i + 1}'

        # For G matrix
        G[non_pumped + i] = '-S_puG_pu(Q_pu)'


        if pipes[non_pumped + i]['start'][0] == 'J':
            junction_number = int(pipes[non_pumped + i]['start'][1:])
            A[non_pumped + i][(3 * non_pumped) + pumped + (junction_number - 1)] = f'-s{non_pumped + i + 1}'
        if pipes[non_pumped + i]['start'][0] == 'R':
            reservoir_number = int(pipes[non_pumped + i]['start'][1:])
            A[non_pumped  + i][(3 * non_pumped) + pumped + junctions + (reservoir_number - 1)] = f'-s{non_pumped + i + 1}'
        if pipes[non_pumped + i]['start'][0] == 'T':
            tank_number = int(pipes[non_pumped + i]['start'][1:])
            A[non_pumped  + i][
                (3 * non_pumped) + pumped + junctions + reservoirs + (tank_number - 1)] = f'-s{non_pumped + i + 1}'

        # Add coeffecients for pumped pipe end Sides connections
        if pipes[non_pumped + i]['end'][0] == 'J':
            junction_number = int(pipes[non_pumped + i]['end'][1:])
            A[non_pumped + i][(3 * non_pumped) + pumped + (junction_number - 1)] = f's{non_pumped + i + 1}'
        if pipes[non_pumped + i]['end'][0] == 'R':
            reservoir_number = int(pipes[non_pumped + i]['end'][1:])
            A[non_pumped + i][(3 * non_pumped) + pumped + junctions + (reservoir_number - 1)] = f's{non_pumped + i + 1}'
        if pipes[non_pumped + i]['end'][0] == 'T':
            tank_number = int(pipes[non_pumped +i]['end'][1:])
            A[non_pumped + i][
                (3 * non_pumped) + pumped + junctions + reservoirs + (tank_number - 1)] = f's{non_pumped + i + 1}'


    # Add coeffecients for pipe end Sides
    for i in range(non_pumped):
        A[non_pumped + pumped + i][i] = f'1-s{i+1}'
        A[non_pumped + pumped + i][(2*non_pumped) + pumped + i] = f's{i+1}'

        # Add coeffecients for pipe end Sides connections
        if pipes[i]['end'][0] == 'J':
            junction_number = int(pipes[i]['end'][1:])
            A[non_pumped + pumped + i][(3 * non_pumped) + pumped + (junction_number - 1)] = f'-s{i + 1}'
        if pipes[i]['end'][0] == 'R':
            reservoir_number = int(pipes[i]['end'][1:])
            A[non_pumped + pumped + i][(3 * non_pumped) + pumped + junctions + (reservoir_number - 1)] = f'-s{i + 1}'
        if pipes[i]['end'][0] == 'T':
            tank_number = int(pipes[i]['end'][1:])
            A[non_pumped + pumped + i][(3 * non_pumped) + pumped + junctions + reservoirs + (tank_number - 1)] = f'-s{i + 1}'


    # Add coefficients for pipe start Sides
    for i in range(non_pumped):
        A[2*non_pumped + pumped + i][non_pumped + pumped + i] = "1"


        if pipes[i]['start'][0] == 'J':
            junction_number = int(pipes[i]['start'][1:])
            A[2*non_pumped + pumped + i][(3 * non_pumped) + pumped + (junction_number - 1)] = "-1"

        if pipes[i]['start'][0] == 'R':
            reservoir_number = int(pipes[i]['start'][1:])
            A[2*non_pumped + pumped + i][(3 * non_pumped) + pumped + junctions + (reservoir_number - 1)] = "-1"

        if pipes[i]['start'][0] == 'T':
            tank_number = int(pipes[i]['start'][1:])
            A[2*non_pumped + pumped + i][(3 * non_pumped) + pumped + junctions + reservoirs + (tank_number - 1)] = "-1"

    # Add coefficients for junctions (it's adjacency for junctions)
    for i, node in enumerate(nodes):
        if node['type'] == 'junction':
            junction_number = int(node['id'][1:])
            for pipe in pipes:
                if pipe['start'] == node['id']:
                    pipe_number = int(pipe['id'][1:])
                    A[(3 * non_pumped) + pumped + (junction_number - 1)][ pipe_number - 1] = "-1"
                if pipe['end'] == node['id']:
                    pipe_number = int(pipe['id'][1:])
                    A[(3 * non_pumped) + pumped + (junction_number - 1)][ pipe_number - 1] = "1"


    #Add 1s for Reservoirs
    for i in range(reservoirs):
        reservoir_idx = 3*non_pumped + pumped + junctions + i
        A[reservoir_idx][reservoir_idx] = '1'

        # Add corresponding P constant to F matrix
        F[reservoir_idx] = f'-P_{chr(97 + i)}'  # P_a, P_b, etc.

    # Add 1s for Tanks
    for pipe in pipes:
        if pipe['start'][0] == 'T':
            tank_number = int(pipe['start'][1:])
            pipe_number = int(pipe['id'][1:])
            A[3 * non_pumped + pumped + junctions + reservoirs + (tank_number - 1)][pipe_number - 1] = '1'




    return {
        'A_matrix': A,
        'E_matrix': E,
        'F_matrix': F,
        'G_matrix': G,
        'components': {
            'non_pumped': non_pumped,
            'pumped': pumped,
            'junctions': junctions,
            'reservoirs': reservoirs,
            'tanks': tanks
        },
        'K_dimension': K
    }

def display_e_matrix(E, components):
    """Display matrix with complete flow labels"""
    labels = []

    # 1. Flow from all pipes
    for i in range(components['non_pumped']):
        labels.append(f"PipeFlow_{i + 1}")

    # 2. Flow from Pumped pipes
    for i in range(components['pumped']):
        labels.append(f"PumpedFlow_{i + 1}")

    # 3. Non-pumped pipes (2 labels per pipe)
    for i in range(components['non_pumped']):
        labels.append(f"Pipe_{i + 1}_L")
        labels.append(f"Pipe_{i + 1}_R")

    # 4. Junctions
    labels += [f"Junction_{i + 1}" for i in range(components['junctions'])]

    # 5. Reservoirs
    labels += [f"Reservoir_{i + 1}" for i in range(components['reservoirs'])]

    # 6. Tanks (last rows)
    labels += [f"Tank_{i + 1}" for i in range(components['tanks'])]

    # Verify label count matches matrix dimension
    assert len(labels) == len(E), f"Labels: {len(labels)} vs Matrix: {len(E)}"

    # Build matrix display
    matrix_str = f"Matrix Dimension: {len(E)}x{len(E[0])}\n\n"
    header = "Row Label".ljust(20) + " ".join(f"{label:<10}" for label in labels)
    matrix_str += header + "\n"

    for row_idx, row in enumerate(E):
        matrix_str += f"{labels[row_idx]:<20}"
        matrix_str += " ".join(f"{'[1]' if x else ' 0 '}" for x in row)
        matrix_str += "\n"

    return matrix_str

def display_A_matrix(A, components):
    """Display A matrix with correct labels matching K dimension"""
    labels = []

    # 1. Flow Non-pumped pipes
    for i in range(components['non_pumped']):
        labels.append(f"Pipe_{i + 1}_Flow")

    # 2. Flow pumped pipes
    for i in range(components['pumped']):
        labels.append(f"Pumped_{i + 1}_Flow")


    # 3. Non-pumped pipes Left
    for i in range(components['non_pumped']):
        labels.append(f"Pipe_{i + 1}_L")

    # 4. Non-pumped pipes Right
    for i in range(components['non_pumped']):
        labels.append(f"Pipe_{i + 1}_R")

    # 5. Junctions
    labels += [f"Junction_{i + 1}" for i in range(components['junctions'])]

    # 6. Reservoirs
    labels += [f"Reservoir_{i + 1}" for i in range(components['reservoirs'])]

    # 7. Tanks (last rows)
    labels += [f"Tank_{i + 1}" for i in range(components['tanks'])]

    # Verify label count matches matrix dimension
    assert len(labels) == len(A), (
        f"Label count ({len(labels)}) ≠ matrix dimension ({len(A)})"
        f"\nComponents: {components}"
    )

    # Build matrix display
    matrix_str = f"A Matrix Dimension: {len(A)}x{len(A[0])}\n\n"
    header = "Row Label".ljust(20) + " ".join(f"{label:<10}" for label in labels)
    matrix_str += header + "\n"

    for row_idx, row in enumerate(A):
        matrix_str += f"{labels[row_idx]:<20}"
        matrix_str += " ".join(f"{val:<10}" for val in row)
        matrix_str += "\n"

    return matrix_str


def display_F_matrix(F, components):
    """Display F matrix with reservoir constants"""
    labels = []

    # Generate labels using same logic as E matrix
    labels += [f"PipeFlow_{i + 1}" for i in range(components['non_pumped'])]
    labels += [f"PumpedFlow_{i + 1}" for i in range(components['pumped'])]
    for i in range(components['non_pumped']):
        labels += [f"Pipe_{i + 1}_L", f"Pipe_{i + 1}_R"]
    labels += [f"Junction_{i + 1}" for i in range(components['junctions'])]
    labels += [f"Reservoir_{i + 1}" for i in range(components['reservoirs'])]
    labels += [f"Tank_{i + 1}" for i in range(components['tanks'])]

    # Build matrix display
    matrix_str = f"F Matrix Dimension: {len(F)}x1\n\n"
    for row_idx, value in enumerate(F):
        matrix_str += f"{labels[row_idx]:<20} {value}\n"

    return matrix_str

def display_G_matrix(G, components):
    """Display G matrix with reservoir constants"""
    labels = []

    labels += [f"PipeFlow_{i + 1}" for i in range(components['non_pumped'])]
    labels += [f"PumpedFlow_{i + 1}" for i in range(components['pumped'])]
    for i in range(components['non_pumped']):
        labels += [f"Pipe_{i + 1}_L", f"Pipe_{i + 1}_R"]
    labels += [f"Junction_{i + 1}" for i in range(components['junctions'])]
    labels += [f"Reservoir_{i + 1}" for i in range(components['reservoirs'])]
    labels += [f"Tank_{i + 1}" for i in range(components['tanks'])]

    #Build matrix display
    matrix_str = f"G Matrix Dimension: {len(G)}x1\n\n"
    for row_idx, value in enumerate(G):
        matrix_str += f"{labels[row_idx]:<20} {value}\n"

    return matrix_str

def visualize_graph():
    G = nx.Graph()
    node_colors = {
        'junction': 'skyblue',
        'tank': 'gold',
        'reservoir': 'salmon'
    }

    # Add nodes
    for node in st.session_state.nodes:
        G.add_node(node['id'], type=node['type'])

    # Add edges with pump information
    for pipe in st.session_state.pipes:
        edge_attrs = {}
        if pipe['has_pump']:
            edge_attrs['style'] = 'dashed'
            edge_attrs['label'] = f"Pump ({pipe['pump_power']}kW)"
        G.add_edge(pipe['start'], pipe['end'], **edge_attrs)

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw nodes
    for node_type, color in node_colors.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[n['id'] for n in st.session_state.nodes if n['type'] == node_type],
                               node_color=color,
                               node_size=1000,
                               ax=ax)

    # Draw regular pipes
    regular_pipes = [(u, v) for u, v, d in G.edges(data=True) if 'style' not in d]
    nx.draw_networkx_edges(G, pos, edgelist=regular_pipes, width=2, ax=ax)

    # Draw pumped pipes
    pumped_pipes = [(u, v) for u, v, d in G.edges(data=True) if 'style' in d]
    nx.draw_networkx_edges(G, pos, edgelist=pumped_pipes, width=2, style='dashed', ax=ax)

    # Add labels and legend
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    # Create legend
    legend_elements = [
        Circle(0, color='skyblue', label='Junction'),
        Circle(0, color='gold', label='Tank'),
        Circle(0, color='salmon', label='Reservoir'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Pump')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.title("Water Network Graph")
    st.pyplot(fig)
# Add to main() function
def main():
    # Node creation
    with st.expander("🏗️ Add Node", expanded=True):
        node_type = st.selectbox("Node Type", ["Junction", "Tank", "Reservoir"])
        if st.button("Add Node"):
            new_node = {
                'id': generate_node_id(node_type.lower()),
                'type': node_type.lower()
            }
            if node_type.lower() == 'tank':
                new_node['capacity'] = 1000  # Default capacity
            elif node_type.lower() == 'reservoir':
                new_node['elevation'] = 50  # Default elevation

            st.session_state.nodes.append(new_node)
            st.success(f"Added {new_node['id']} ({node_type})")

    # Pipe creation
    with st.expander("🕳️ Add Pipe"):
        node_ids = [n['id'] for n in st.session_state.nodes]
        col1, col2 = st.columns(2)
        with col1:
            start_node = st.selectbox("Start Node", node_ids)
        with col2:
            end_node = st.selectbox("End Node", node_ids)

        has_pump = st.checkbox("Contains Pump")
        pump_power = st.number_input("Pump Power (kW)", min_value=0.0) if has_pump else 0.0

        if st.button("Add Pipe"):
            if start_node == end_node:
                st.error("Cannot connect a node to itself!")
            else:
                new_pipe = {
                    'id': generate_pipe_id(),
                    'start': start_node,
                    'end': end_node,
                    'has_pump': has_pump,
                    'pump_power': pump_power
                }
                st.session_state.pipes.append(new_pipe)
                st.success(f"Added {new_pipe['id']} between {start_node} and {end_node}")

    # Network visualization
    st.subheader("Network Visualization")
    if st.session_state.nodes:
        visualize_network()
    else:
        st.info("No nodes added yet")

    # Raw data display
    with st.expander("📁 View Raw Data"):
        st.write("Nodes:", st.session_state.nodes)
        st.write("Pipes:", st.session_state.pipes)

    with st.expander("🧮 Network Matrices"):
        if st.session_state.nodes and st.session_state.pipes:
            matrices = create_network_matrices(st.session_state.nodes, st.session_state.pipes)

            st.subheader("E Matrix Structure")
            st.write(f"Matrix Dimension (K x K): {matrices['K_dimension']} x {matrices['K_dimension']}")

            # Visualize E matrix pattern
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(matrices['E_matrix'], cmap='binary')
            ax.set_title("E Matrix Pattern (1=active, 0=inactive)")
            st.pyplot(fig)

            st.write("Network Parameters:", matrices['components'])
            st.write(pd.DataFrame(matrices['E_matrix']))



    with st.expander("🧮 Network Matrices Detailed"):
        if st.session_state.nodes and st.session_state.pipes:
            matrices = create_network_matrices(st.session_state.nodes, st.session_state.pipes)

            st.subheader("Enhanced E Matrix Display")

            # Text-based display with labels
            st.text(display_e_matrix(matrices['E_matrix'], matrices['components']))

            # Color version for terminals (optional)
            st.code(display_e_matrix(matrices['E_matrix'], matrices['components']), language='ansi')

            # Simplified LaTeX version with labels
            latex_str = r"\begin{bmatrix}" + "\n"
            for row in matrices['E_matrix']:
                latex_str += " & ".join([r"\mathbf{1}" if x else "0" for x in row]) + r" \\" + "\n"
            latex_str += r"\end{bmatrix}"
            st.latex(latex_str)

    with st.expander("🧮 A Matrix"):
        if st.session_state.nodes and st.session_state.pipes:
            matrices = create_network_matrices(st.session_state.nodes, st.session_state.pipes)

            st.subheader("A Matrix Structure")
            st.write(f"Matrix Dimension (K x K): {matrices['K_dimension']} x {matrices['K_dimension']}")

            # Visualize A matrix pattern
            fig, ax = plt.subplots(figsize=(8, 8))
            numeric_A = [[1 if cell != '0' else 0 for cell in row] for row in matrices['A_matrix']]
            ax.imshow(numeric_A, cmap='binary')
            ax.set_title("A Matrix Pattern (Coefficient=1, 0=inactive)")
            st.pyplot(fig)

            st.subheader("Detailed A Matrix")
            st.text(display_A_matrix(matrices['A_matrix'], matrices['components']))

            # LaTeX version
            latex_str = r"\begin{bmatrix}"
            for row in matrices['A_matrix']:
                latex_str += " & ".join(row) + r" \\ "
            latex_str += r"\end{bmatrix}"
            st.latex(latex_str)

            st.write("Network Parameters:", matrices['components'])
            st.write(pd.DataFrame(matrices['A_matrix']))

    with st.expander("🧮 F Matrix"):
        if st.session_state.nodes and st.session_state.pipes:
            matrices = create_network_matrices(st.session_state.nodes, st.session_state.pipes)

            # Add F matrix display
            st.subheader("F Matrix Structure")
            st.write(f"Matrix Dimension (K x 1): {matrices['K_dimension']} x 1")
            st.text(display_F_matrix(matrices['F_matrix'], matrices['components']))

            # Add LaTeX for F matrix
            latex_str = r"\begin{bmatrix}"
            latex_str += r" \\ ".join(matrices['F_matrix'])
            latex_str += r"\end{bmatrix}"
            st.latex(latex_str)

    with st.expander("🧮 G Matrix"):
        if st.session_state.nodes and st.session_state.pipes:
            matrices = create_network_matrices(st.session_state.nodes, st.session_state.pipes)

            # Add G matrix display
            st.subheader("G Matrix Structure")
            st.write(f"Matrix Dimension (K x 1): {matrices['K_dimension']} x 1")
            st.text(display_G_matrix(matrices['G_matrix'], matrices['components']))

            # Correct subscript
            matrices['G_matrix'] = [x.replace('pu', r'{\text{pu}}') for x in matrices['G_matrix']]

            # Add LaTeX for G matrix
            latex_str = r"\begin{bmatrix}"
            latex_str += r" \\ ".join(matrices['G_matrix'])
            latex_str += r"\end{bmatrix}"
            st.latex(latex_str)

    #Add footer text
    st.markdown("---")
    st.write("© 2025 Streamlit Water Network Demo | [GitHub Repository](Abdullah-Afzal-00/Water_Networks_App)")
    st.write("Built by [Abdullah Afzal](https://github.com/Abdullah-Afzal-00/)")


if __name__ == "__main__":
    main()