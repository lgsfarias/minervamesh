import numpy as np
import meshio
import plotly.graph_objects as go
from pyscript import when, display
from js import document, FileReader

# Variável global para armazenar os dados da malha
mesh_data = None

def build_matrices(X, Y, IEN):
    """Constrói as matrizes K, M, Gx, Gy do MEF."""
    npoints = len(X)
    ne = len(IEN)
    
    # Inicializar matrizes
    K = np.zeros((npoints, npoints), dtype='float')
    M = np.zeros((npoints, npoints), dtype='float')
    Gx = np.zeros((npoints, npoints), dtype='float')
    Gy = np.zeros((npoints, npoints), dtype='float')
    
    # Loop dos elementos da malha
    for e in range(ne):
        v = IEN[e]
        
        # Área do elemento
        det = X[v[2]]*(Y[v[0]]-Y[v[1]]) + X[v[0]]*(Y[v[1]]-Y[v[2]]) + X[v[1]]*(-Y[v[0]]+Y[v[2]])
        area = det/2.0
        
        # Matrizes do elemento linear
        m = (area/12.0) * np.array([[2.0, 1.0, 1.0],
                                   [1.0, 2.0, 1.0],
                                   [1.0, 1.0, 2.0]])
        
        # Coeficientes do elemento triangular linear
        b1 = Y[v[1]]-Y[v[2]]
        b2 = Y[v[2]]-Y[v[0]]
        b3 = Y[v[0]]-Y[v[1]]
        
        c1 = X[v[2]]-X[v[1]]
        c2 = X[v[0]]-X[v[2]]
        c3 = X[v[1]]-X[v[0]]
        
        # Matriz do gradiente
        B = (1.0/(2.0*area)) * np.array([[b1, b2, b3],
                                        [c1, c2, c3]])
        BT = B.transpose()
        
        # Matriz de rigidez do elemento
        kele = area * np.dot(BT, B)
        
        # Matrizes de gradiente
        gxele = (1.0/6.0) * np.array([[b1, b2, b3],
                                     [b1, b2, b3],
                                     [b1, b2, b3]])
        gyele = (1.0/6.0) * np.array([[c1, c2, c3],
                                     [c1, c2, c3],
                                     [c1, c2, c3]])
        
        # Montagem (assembling) das matrizes
        for i in range(3):
            ii = IEN[e, i]
            for j in range(3):
                jj = IEN[e, j]
                K[ii, jj] += kele[i, j]
                M[ii, jj] += m[i, j]
                Gx[ii, jj] += gxele[i, j]
                Gy[ii, jj] += gyele[i, j]
    
    return K, M, Gx, Gy

def identify_boundary_nodes(X, Y, eps=1e-6):
    """Identifica nós de contorno baseado na posição."""
    npoints = len(X)
    boundary_nodes = []
    
    for i in range(npoints):
        # Contorno inferior (y = 0)
        if abs(Y[i]) < eps:
            boundary_nodes.append(i)
        # Contorno superior (y = Y.max())
        elif abs(Y[i] - Y.max()) < eps:
            boundary_nodes.append(i)
        # Contorno esquerdo (x = 0)
        elif abs(X[i]) < eps:
            boundary_nodes.append(i)
        # Contorno direito (x = X.max())
        elif abs(X[i] - X.max()) < eps:
            boundary_nodes.append(i)
    
    return np.array(boundary_nodes)

def solve_stream_function(X, Y, IEN, u_in=1.0):
    """Resolve a equação da função corrente."""
    npoints = len(X)
    
    # Construir matrizes
    K, M, Gx, Gy = build_matrices(X, Y, IEN)
    
    # Identificar nós de contorno
    boundary_nodes = identify_boundary_nodes(X, Y)
    
    # Copiar matriz K
    mat = K.copy()
    
    # Impor condições de contorno
    for i in boundary_nodes:
        mat[i, :] = 0.0
        mat[i, i] = 1.0
    
    # Vetor do lado direito
    Z = np.zeros((npoints, 1), dtype='float')
    
    # Aplicar condições de contorno na função corrente
    for i in boundary_nodes:
        if abs(Y[i]) < 1e-6:  # Contorno inferior
            Z[i] = 0.0
        elif abs(Y[i] - Y.max()) < 1e-6:  # Contorno superior
            Z[i] = Y[i]
        elif abs(X[i]) < 1e-6:  # Contorno esquerdo
            Z[i] = Y[i]
        else:  # Contorno direito
            Z[i] = Y[i]
    
    # Resolver sistema linear
    psi = np.linalg.solve(mat, Z)
    
    return psi.flatten()

def solve_stream_vorticity(X, Y, IEN, nu=1.0, dt=0.001, max_iter=100, u_in=1.0):
    """Resolve o sistema função corrente-vorticidade."""
    npoints = len(X)
    
    # Construir matrizes
    K, M, Gx, Gy = build_matrices(X, Y, IEN)
    
    # Identificar nós de contorno
    boundary_nodes = identify_boundary_nodes(X, Y)
    
    # Inicializar velocidades
    vx = np.zeros(npoints, dtype='float')
    vy = np.zeros(npoints, dtype='float')
    
    # Impor condições de contorno iniciais
    for i in boundary_nodes:
        if abs(Y[i]) < 1e-6:  # Entrada (contorno inferior)
            vx[i] = u_in
            vy[i] = 0.0
        else:  # Paredes
            vx[i] = 0.0
            vy[i] = 0.0
    
    # Loop no tempo
    for iteration in range(max_iter):
        # Calcular vorticidade para inclusão no contorno
        b = np.dot(Gx, vy) - np.dot(Gy, vx)
        omega = np.linalg.solve(M, b)
        omega_bc = omega.copy()
        
        # Resolver transporte da vorticidade
        A = (1.0/dt) * M + nu * K
        
        # Termo convectivo v \cdot \nabla \omega
        vgo = vx * np.dot(Gx, omega) + vy * np.dot(Gy, omega)
        
        # Vetor do lado direito para equação de transporte
        b_1 = (1.0/dt) * np.dot(M, omega) - vgo
        
        # Impor condições de contorno para omega
        for i in boundary_nodes:
            A[i, :] = 0.0
            A[i, i] = 1.0
            b_1[i] = omega_bc[i]
        
        # Resolver equação de transporte da vorticidade
        omega = np.linalg.solve(A, b_1)
        
        # Resolver função corrente K*psi = M*omega
        b_2 = np.dot(M, omega)
        A_2 = K.copy()
        
        # Identificar condições de contorno de psi
        psi_bc = np.zeros(npoints, dtype='float')
        psi_boundary = []
        
        for i in boundary_nodes:
            if abs(Y[i]) < 1e-6:  # Entrada
                psi_bc[i] = Y[i]
                psi_boundary.append(i)
            else:  # Paredes
                psi_bc[i] = Y[i]
                psi_boundary.append(i)
        
        # Impor condições de contorno de psi
        for i in psi_boundary:
            A_2[i, :] = 0.0
            A_2[i, i] = 1.0
            b_2[i] = psi_bc[i]
        
        # Resolver função corrente
        psi = np.linalg.solve(A_2, b_2)
        
        # Calcular velocidades M*vx = Gy*psi, M*vy = -Gx*psi
        b_3 = np.dot(Gy, psi)
        vx = np.linalg.solve(M, b_3)
        
        b_4 = np.dot(Gx, psi)
        vy = -np.linalg.solve(M, b_4)
        
        # Reimpor condições de contorno de velocidade
        for i in boundary_nodes:
            if abs(Y[i]) < 1e-6:  # Entrada
                vx[i] = u_in
                vy[i] = 0.0
            else:  # Paredes
                vx[i] = 0.0
                vy[i] = 0.0
    
    return psi, omega, vx, vy

@when("change", "#mesh-file")
def load_mesh(evt=None):
    global mesh_data
    print("Debug: Função load_mesh chamada")  # Debug
    input_elem = document.getElementById("mesh-file")
    file = input_elem.files.item(0)
    if not file:
        document.getElementById("status").textContent = "Selecione um arquivo de malha."
        document.getElementById("status").className = "status error"
        document.getElementById("status").style.display = "block"
        return

    print(f"Debug: Arquivo selecionado: {file.name}")  # Debug

    def onload(e):
        global mesh_data
        print("Debug: Função onload chamada")  # Debug
        content = e.target.result
        print(f"Debug: Arquivo carregado, tamanho: {len(content)} caracteres")  # Debug
        
        try:
            # Salvar o conteúdo como arquivo temporário virtual
            with open("/tmp/uploaded.msh", "w") as f:
                f.write(content)
            
            # Ler malha com meshio
            mesh = meshio.read("/tmp/uploaded.msh")
            X = mesh.points[:, 0]
            Y = mesh.points[:, 1]
            IEN = mesh.cells_dict["triangle"]
            # Identificar elementos de linha (contornos)
            if "line" in mesh.cells_dict:
                lines = mesh.cells_dict["line"]
            else:
                lines = np.array([])
            # Identificar nós de contorno
            boundary_nodes = set(lines.flatten()) if len(lines) > 0 else set()
            # Separar contorno externo e buraco pelo bounding box
            ext_nodes = set()
            hole_nodes = set()
            if len(boundary_nodes) > 0:
                # Calcular bounding box de todos os nós
                min_x, max_x = X.min(), X.max()
                min_y, max_y = Y.min(), Y.max()
                for n in boundary_nodes:
                    if abs(X[n] - min_x) < 1e-8 or abs(X[n] - max_x) < 1e-8 or abs(Y[n] - min_y) < 1e-8 or abs(Y[n] - max_y) < 1e-8:
                        ext_nodes.add(n)
                    else:
                        hole_nodes.add(n)
            mesh_data = {
                'X': X,
                'Y': Y,
                'IEN': IEN,
                'domain': {'L': float(X.max()), 'H': float(Y.max())},
                'ext_nodes': np.array(list(ext_nodes)),
                'hole_nodes': np.array(list(hole_nodes))
            }
            print(f"Debug: ext_nodes: {len(ext_nodes)}, hole_nodes: {len(hole_nodes)}")
            # Plot da malha
            plot_mesh(mesh_data)
            # Atualizar informações
            document.getElementById("mesh-info").style.display = "block"
            document.getElementById("mesh-name").textContent = file.name
            document.getElementById("mesh-nodes").textContent = str(len(X))
            document.getElementById("mesh-elements").textContent = str(len(IEN))
            document.getElementById("status").textContent = "Malha carregada! Pronta para simulação."
            document.getElementById("status").className = "status success"
            document.getElementById("status").style.display = "block"
        except Exception as e:
            print(f"Debug: Erro ao carregar malha: {str(e)}")  # Debug
            document.getElementById("status").textContent = f"Erro ao carregar malha: {str(e)}"
            document.getElementById("status").className = "status error"
            document.getElementById("status").style.display = "block"
    reader = FileReader.new()
    reader.onload = onload
    reader.readAsText(file)

def plot_mesh(mesh_data):
    """Plota a malha importada."""
    X = mesh_data['X']
    Y = mesh_data['Y']
    IEN = mesh_data['IEN']
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar triângulos da malha
    for tri in IEN:
        x = [X[tri[0]], X[tri[1]], X[tri[2]], X[tri[0]]]
        y = [Y[tri[0]], Y[tri[1]], Y[tri[2]], Y[tri[0]]]
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='lines', 
            line=dict(color='blue', width=1), 
            showlegend=False
        ))
    
    # Adicionar pontos da malha
    fig.add_trace(go.Scatter(
        x=X, y=Y,
        mode='markers',
        marker=dict(color='red', size=3),
        name='Nós',
        hovertemplate='<b>Nó</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Calcular dimensões mantendo proporção
    L = mesh_data['domain']['L']
    H = mesh_data['domain']['H']
    width = 800
    aspect_ratio = L / H
    height = int(width / aspect_ratio)
    height = max(height, 400)
    
    # Configurar layout
    fig.update_layout(
        title="Malha Importada",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        height=height,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, H]),
        xaxis_range=[0, L]
    )
    
    display(fig, target="psi-plot")

@when("click", "#run-sim-btn")
def run_simulation(evt=None):
    global mesh_data
    print(f"Debug: mesh_data = {mesh_data}")  # Debug
    if mesh_data is None:
        document.getElementById("status").textContent = "Carregue uma malha primeiro!"
        document.getElementById("status").className = "status error"
        document.getElementById("status").style.display = "block"
        return
    
    # Ler parâmetros
    u_in = float(document.getElementById("u_in").value)
    nu = float(document.getElementById("nu").value)
    dt = float(document.getElementById("dt").value)
    max_iter = int(document.getElementById("max_iter").value)
    
    document.getElementById("loading").style.display = "block"
    document.getElementById("status").style.display = "none"
    
    try:
        print(f"Debug: Executando simulação com {len(mesh_data['X'])} nós")  # Debug
        # Executar simulação
        psi, omega, vx, vy = solve_stream_vorticity(
            mesh_data['X'], 
            mesh_data['Y'], 
            mesh_data['IEN'], 
            nu=nu, 
            dt=dt, 
            max_iter=max_iter, 
            u_in=u_in
        )
        
        document.getElementById("loading").style.display = "none"
        
        # Plotar resultados
        plot_results(mesh_data, psi, omega, vx, vy, max_iter, dt)
        
        # Atualizar informações
        document.getElementById("simulation-info").style.display = "block"
        document.getElementById("reynolds").textContent = f"{u_in * mesh_data['domain']['L'] / nu:.2f}"
        document.getElementById("sim-time").textContent = f"{max_iter * dt:.4f} s"
        document.getElementById("max-velocity").textContent = f"{np.sqrt((vx**2 + vy**2).max()):.4f}"
        document.getElementById("max-vorticity").textContent = f"{np.abs(omega).max():.4f}"
        
        document.getElementById("status").textContent = "Simulação executada com sucesso!"
        document.getElementById("status").className = "status success"
        document.getElementById("status").style.display = "block"
        
    except Exception as e:
        document.getElementById("loading").style.display = "none"
        document.getElementById("status").textContent = f"Erro na simulação: {str(e)}"
        document.getElementById("status").className = "status error"
        document.getElementById("status").style.display = "block"

def plot_results(mesh_data, psi, omega, vx, vy, max_iter, dt):
    """Plota os resultados da simulação."""
    X = mesh_data['X']
    Y = mesh_data['Y']
    IEN = mesh_data['IEN']
    L = mesh_data['domain']['L']
    H = mesh_data['domain']['H']
    
    # Calcular dimensões mantendo proporção
    width = 800
    aspect_ratio = L / H
    height = int(width / aspect_ratio)
    height = max(height, 400)
    
    # Função corrente
    fig_psi = go.Figure(data=[
        go.Scatter(
            x=X, y=Y,
            mode='markers',
            marker=dict(
                color=psi, 
                colorscale='Viridis', 
                size=6, 
                colorbar=dict(title='ψ')
            ),
            name='Função Corrente',
            hovertemplate='<b>Função Corrente</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>ψ: %{marker.color:.4f}<extra></extra>'
        )
    ])
    fig_psi.update_layout(
        title="Função Corrente",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        height=height,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, H]),
        xaxis_range=[0, L]
    )
    display(fig_psi, target="psi-plot")
    
    # Vorticidade
    fig_omega = go.Figure(data=[
        go.Scatter(
            x=X, y=Y,
            mode='markers',
            marker=dict(
                color=omega, 
                colorscale='RdBu', 
                size=6, 
                colorbar=dict(title='ω')
            ),
            name='Vorticidade',
            hovertemplate='<b>Vorticidade</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>ω: %{marker.color:.4f}<extra></extra>'
        )
    ])
    fig_omega.update_layout(
        title="Vorticidade",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        height=height,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, H]),
        xaxis_range=[0, L]
    )
    display(fig_omega, target="omega-plot")
    
    # Campo de velocidades
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    fig_vel = go.Figure(data=[
        go.Scatter(
            x=X, y=Y,
            mode='markers',
            marker=dict(
                color=velocity_magnitude,
                colorscale='Blues',
                size=6,
                colorbar=dict(title='|V|')
            ),
            name='Velocidade',
            hovertemplate='<b>Velocidade</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>|V|: %{marker.color:.4f}<extra></extra>'
        )
    ])
    fig_vel.update_layout(
        title="Campo de Velocidades",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        height=height,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, H]),
        xaxis_range=[0, L]
    )
    display(fig_vel, target="velocity-plot")
    
    # Linhas de corrente
    fig_stream = go.Figure()
    for tri in IEN:
        x = [X[tri[0]], X[tri[1]], X[tri[2]], X[tri[0]]]
        y = [Y[tri[0]], Y[tri[1]], Y[tri[2]], Y[tri[0]]]
        fig_stream.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))
    fig_stream.add_trace(go.Scatter(
        x=X, y=Y,
        mode='markers',
        marker=dict(color=psi, colorscale='Viridis', size=6),
        name='Linhas de Corrente',
        hovertemplate='<b>Linhas de Corrente</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>ψ: %{marker.color:.4f}<extra></extra>'
    ))
    fig_stream.update_layout(
        title="Linhas de Corrente",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        height=height,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, H]),
        xaxis_range=[0, L]
    )
    display(fig_stream, target="streamlines-plot") 