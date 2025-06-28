import numpy as np
import json
import meshio
from pyscript import when, display
import plotly.graph_objects as go
from js import console, document, Blob
import tempfile
import os
from scipy.spatial import Delaunay
import js


class HoleFactory:
    """Fábrica de furos que define os tipos disponíveis."""
    
    hole_types = {
        "circle": {
            "params": [
                {"name": "radius", "label": "Raio do Furo", "default": 0.1, "min": 0.05, "max": 0.3}
            ]
        },
        "square": {
            "params": [
                {"name": "width", "label": "Largura do Furo", "default": 0.1, "min": 0.05, "max": 0.3},
                {"name": "height", "label": "Altura do Furo", "default": 0.1, "min": 0.05, "max": 0.3}
            ]
        },
        "elipse": {
            "params": [
                {"name": "rx", "label": "Raio X da Elipse", "default": 0.15, "min": 0.05, "max": 0.3},
                {"name": "ry", "label": "Raio Y da Elipse", "default": 0.1, "min": 0.05, "max": 0.3}
            ]
        }
    }

class MeshGenerator:
    """Classe responsável por gerar a malha usando triangulação de Delaunay."""
    
    def __init__(self, lc, L, H, hole_type, hole_params):
        self.lc = lc
        self.L = L
        self.H = H
        self.hole_type = hole_type
        self.hole_params = hole_params
    
    def generate(self):
        """Gera a malha com o furo especificado usando triangulação de Delaunay."""
        # Gerar pontos do domínio retangular
        nx = int(self.L / self.lc) + 1
        ny = int(self.H / self.lc) + 1
        
        x = np.linspace(0, self.L, nx)
        y = np.linspace(0, self.H, ny)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.flatten(), Y.flatten()])
        
        # Remover pontos dentro do furo
        hole_x, hole_y = self.L / 2, self.H / 2
        mask = np.ones(len(points), dtype=bool)
        
        if self.hole_type == "circle":
            radius = self.hole_params["radius"]
            dist = np.sqrt((points[:, 0] - hole_x)**2 + (points[:, 1] - hole_y)**2)
            mask = dist > radius
            
        elif self.hole_type == "square":
            width = self.hole_params["width"]
            height = self.hole_params["height"]
            x_mask = (points[:, 0] < hole_x - width/2) | (points[:, 0] > hole_x + width/2)
            y_mask = (points[:, 1] < hole_y - height/2) | (points[:, 1] > hole_y + height/2)
            mask = x_mask | y_mask
            
        elif self.hole_type == "elipse":
            rx = self.hole_params["rx"]
            ry = self.hole_params["ry"]
            dist = ((points[:, 0] - hole_x)**2 / rx**2) + ((points[:, 1] - hole_y)**2 / ry**2)
            mask = dist > 1
        
        # Filtrar pontos
        points = points[mask]
        
        # Adicionar pontos no contorno do furo para melhor qualidade da malha
        boundary_points = self._generate_boundary_points(hole_x, hole_y)
        points = np.vstack([points, boundary_points])
        
        # Gerar triangulação de Delaunay
        tri = Delaunay(points)
        
        # Função para verificar se um ponto está dentro do furo
        def is_in_hole(centroid, hole_type, hole_x, hole_y, hole_params):
            if hole_type == "circle":
                radius = hole_params["radius"]
                dist = np.sqrt((centroid[0] - hole_x)**2 + (centroid[1] - hole_y)**2)
                return dist < radius
            elif hole_type == "square":
                width = hole_params["width"]
                height = hole_params["height"]
                return (hole_x - width/2 < centroid[0] < hole_x + width/2) and (hole_y - height/2 < centroid[1] < hole_y + height/2)
            elif hole_type == "elipse":
                rx = hole_params["rx"]
                ry = hole_params["ry"]
                dist = ((centroid[0] - hole_x)**2 / rx**2) + ((centroid[1] - hole_y)**2 / ry**2)
                return dist < 1
            return False

        # Filtrar triângulos cujo centróide está dentro do furo
        filtered_triangles = []
        for triangle in tri.simplices:
            pts = points[triangle]
            centroid = np.mean(pts, axis=0)
            if not is_in_hole(centroid, self.hole_type, hole_x, hole_y, self.hole_params):
                filtered_triangles.append(triangle)
        filtered_triangles = np.array(filtered_triangles)
        
        # Criar arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(suffix='.msh', delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Salvar como arquivo .msh simples
        with open(temp_filename, 'w') as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
            f.write(f"$Nodes\n{len(points)}\n")
            for i, point in enumerate(points):
                f.write(f"{i+1} {point[0]:.6f} {point[1]:.6f} 0.0\n")
            f.write("$EndNodes\n")
            f.write(f"$Elements\n{len(filtered_triangles)}\n")
            for i, triangle in enumerate(filtered_triangles):
                f.write(f"{i+1} 2 2 1 1 {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
            f.write("$EndElements\n")
        
        return temp_filename
    
    def _generate_boundary_points(self, hole_x, hole_y):
        """Gera pontos no contorno do furo."""
        boundary_points = []
        
        if self.hole_type == "circle":
            radius = self.hole_params["radius"]
            n_points = max(20, int(2 * np.pi * radius / self.lc))
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            for angle in angles:
                x = hole_x + radius * np.cos(angle)
                y = hole_y + radius * np.sin(angle)
                boundary_points.append([x, y])
                
        elif self.hole_type == "square":
            width = self.hole_params["width"]
            height = self.hole_params["height"]
            # Pontos nas bordas do quadrado
            x_left = hole_x - width/2
            x_right = hole_x + width/2
            y_bottom = hole_y - height/2
            y_top = hole_y + height/2
            
            # Bordas horizontais
            n_h = max(10, int(width / self.lc))
            x_h = np.linspace(x_left, x_right, n_h)
            for x in x_h:
                boundary_points.extend([[x, y_bottom], [x, y_top]])
            
            # Bordas verticais
            n_v = max(10, int(height / self.lc))
            y_v = np.linspace(y_bottom, y_top, n_v)
            for y in y_v:
                boundary_points.extend([[x_left, y], [x_right, y]])
                
        elif self.hole_type == "elipse":
            rx = self.hole_params["rx"]
            ry = self.hole_params["ry"]
            n_points = max(30, int(2 * np.pi * max(rx, ry) / self.lc))
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            for angle in angles:
                x = hole_x + rx * np.cos(angle)
                y = hole_y + ry * np.sin(angle)
                boundary_points.append([x, y])
        
        return np.array(boundary_points)

def validate_parameters(lc, L, H, hole_type, hole_params):
    """Valida os parâmetros de entrada."""
    errors = []
    
    if lc < 0.01 or lc > 0.2:
        errors.append("Tamanho da malha deve estar entre 0.01 e 0.2")
    
    if L < 0.5 or L > 2.0:
        errors.append("Comprimento deve estar entre 0.5 e 2.0")
    
    if H < 0.2 or H > 1.0:
        errors.append("Altura deve estar entre 0.2 e 1.0")
    
    # Validar parâmetros do furo
    max_size = min(L, H) / 2 - 0.01
    if 'radius' in hole_params and hole_params['radius'] >= max_size:
        errors.append("O raio do furo é muito grande para o domínio!")
    
    return errors

def generate_mesh_data(lc, L, H, hole_type, hole_params):
    """Função principal para gerar dados da malha."""
    # Validar parâmetros
    errors = validate_parameters(lc, L, H, hole_type, hole_params)
    if errors:
        return {"error": errors}
    
    try:
        # Gerar malha
        mesh_generator = MeshGenerator(lc, L, H, hole_type, hole_params)
        mesh_file = mesh_generator.generate()
        
        # Ler malha com meshio
        mesh = meshio.read(mesh_file)
        points = mesh.points[:, :2]  # Apenas coordenadas x, y
        cells = np.array(mesh.cells_dict["triangle"])
        
        # Limpar arquivo temporário
        try:
            os.unlink(mesh_file)
        except:
            pass
        
        return {
            'points': points.tolist(),
            'triangles': cells.tolist(),
            'hole_type': hole_type,
            'hole_params': hole_params,
            'domain': {'L': L, 'H': H},
            'lc': lc
        }
    except Exception as e:
        return {"error": [f"Erro ao gerar malha: {str(e)}"]}

def get_hole_params(hole_type):
    """Obtém os parâmetros do furo baseado no tipo selecionado."""
    hole_params = {}
    
    if hole_type == "circle":
        radius_elem = document.getElementById("hole-radius")
        if radius_elem:
            hole_params["radius"] = float(radius_elem.value)
    
    elif hole_type == "square":
        width_elem = document.getElementById("hole-width")
        height_elem = document.getElementById("hole-height")
        if width_elem and height_elem:
            hole_params["width"] = float(width_elem.value)
            hole_params["height"] = float(height_elem.value)
    
    elif hole_type == "elipse":
        rx_elem = document.getElementById("hole-rx")
        ry_elem = document.getElementById("hole-ry")
        if rx_elem and ry_elem:
            hole_params["rx"] = float(rx_elem.value)
            hole_params["ry"] = float(ry_elem.value)
    
    return hole_params

def update_hole_parameters():
    """Atualiza os campos de parâmetros do furo baseado no tipo selecionado."""
    hole_type = document.getElementById("hole-type").value
    hole_params_div = document.getElementById("hole-params")
    
    if hole_type == "circle":
        hole_params_div.innerHTML = """
            <div class="form-group">
                <label for="hole-radius">Raio do Furo:</label>
                <input type="number" id="hole-radius" value="0.1" step="0.01" min="0.05" max="0.3">
                <div class="range-info">(0.05 - 0.3)</div>
            </div>
        """
    elif hole_type == "square":
        hole_params_div.innerHTML = """
            <div class="form-group">
                <label for="hole-width">Largura do Furo:</label>
                <input type="number" id="hole-width" value="0.1" step="0.01" min="0.05" max="0.3">
                <div class="range-info">(0.05 - 0.3)</div>
            </div>
            <div class="form-group">
                <label for="hole-height">Altura do Furo:</label>
                <input type="number" id="hole-height" value="0.1" step="0.01" min="0.05" max="0.3">
                <div class="range-info">(0.05 - 0.3)</div>
            </div>
        """
    elif hole_type == "elipse":
        hole_params_div.innerHTML = """
            <div class="form-group">
                <label for="hole-rx">Raio X da Elipse:</label>
                <input type="number" id="hole-rx" value="0.15" step="0.01" min="0.05" max="0.3">
                <div class="range-info">(0.05 - 0.3)</div>
            </div>
            <div class="form-group">
                <label for="hole-ry">Raio Y da Elipse:</label>
                <input type="number" id="hole-ry" value="0.1" step="0.01" min="0.05" max="0.3">
                <div class="range-info">(0.05 - 0.3)</div>
            </div>
        """

def plot_mesh(mesh_data):
    """Cria a visualização da malha usando Plotly."""
    points = np.array(mesh_data['points'])
    triangles = np.array(mesh_data['triangles'])
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar pontos da malha (primeiro!)
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers',
        marker=dict(color='red', size=3),
        name='Nós da Malha',
        showlegend=True,
        hovertemplate='<b>Nó</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Adicionar triângulos da malha
    for triangle in triangles:
        x = [points[triangle[0]][0], points[triangle[1]][0], points[triangle[2]][0], points[triangle[0]][0]]
        y = [points[triangle[0]][1], points[triangle[1]][1], points[triangle[2]][1], points[triangle[0]][1]]
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='blue', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Calcular dimensões mantendo proporção
    L = mesh_data['domain']['L']
    H = mesh_data['domain']['H']
    
    # Usar 100% da largura disponível (assumindo que a div tem largura máxima de 800px)
    # Você pode ajustar este valor conforme necessário
    width = 800  # Largura total da div
    aspect_ratio = L / H
    
    # Calcular altura proporcional
    height = int(width / aspect_ratio)
    
    # Garantir altura mínima
    height = max(height, 500)
    
    # Configurar layout
    fig.update_layout(
        title="Visualização da Malha Gerada",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=True,
        height=height,
        showlegend=True,
        hovermode='closest'
    )
    
    # Manter proporção 1:1 nos eixos
    fig.update_xaxes(range=[0, L])
    fig.update_yaxes(range=[0, H])
    
    # Forçar proporção 1:1 nos eixos para evitar distorção
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
        )
    )
    
    return fig

def update_mesh_info(mesh_data):
    """Atualiza as informações da malha na interface."""
    points = np.array(mesh_data['points'])
    triangles = np.array(mesh_data['triangles'])
    
    document.getElementById("num-nodes").textContent = str(len(points))
    document.getElementById("num-elements").textContent = str(len(triangles))
    document.getElementById("min-size").textContent = f"{mesh_data['lc']:.3f}"
    
    # Mostrar a seção de informações
    document.getElementById("mesh-info").style.display = "block"

def show_status(message, status_type="info"):
    """Mostra uma mensagem de status na interface."""
    status_elem = document.getElementById("status")
    status_elem.className = f"status {status_type}"
    status_elem.textContent = message
    status_elem.style.display = "block"

def hide_status():
    """Esconde a mensagem de status."""
    document.getElementById("status").style.display = "none"

def show_loading():
    """Mostra o indicador de carregamento."""
    document.getElementById("loading").style.display = "block"

def hide_loading():
    """Esconde o indicador de carregamento."""
    document.getElementById("loading").style.display = "none"

# Variável global para armazenar a última malha gerada
last_mesh_data = None

@when("click", "#generate-btn")
def generate_mesh():
    """Função principal executada quando o botão 'Gerar Malha' é clicado."""
    global last_mesh_data
    try:
        # Mostrar loading
        show_loading()
        hide_status()
        
        # Coletar parâmetros
        lc = float(document.getElementById("lc").value)
        L = float(document.getElementById("L").value)
        H = float(document.getElementById("H").value)
        hole_type = document.getElementById("hole-type").value
        
        # Obter parâmetros do furo
        hole_params = get_hole_params(hole_type)
        
        # Validar se todos os parâmetros foram obtidos
        if not hole_params:
            show_status("Erro: Parâmetros do furo não encontrados!", "error")
            document.getElementById("export-btn").disabled = True
            return
        
        # Gerar malha
        result = generate_mesh_data(lc, L, H, hole_type, hole_params)
        
        if "error" in result:
            show_status(f"Erro: {'; '.join(result['error'])}", "error")
            document.getElementById("export-btn").disabled = True
            return
        
        # Limpar o container do gráfico antes de exibir o novo
        document.getElementById("mesh-plot").innerHTML = ""
        
        # Plotar malha
        fig = plot_mesh(result)
        display(fig, target="mesh-plot")
        
        # Atualizar informações
        update_mesh_info(result)
        
        # Armazenar a última malha gerada
        last_mesh_data = result
        # Habilitar botão de exportação
        document.getElementById("export-btn").disabled = False
        
        # Mostrar sucesso
        show_status("Malha gerada com sucesso!", "success")
        
    except Exception as e:
        console.error(f"Erro ao gerar malha: {str(e)}")
        show_status(f"Erro inesperado: {str(e)}", "error")
        document.getElementById("export-btn").disabled = True
    
    finally:
        hide_loading()

@when("change", "#hole-type")
def on_hole_type_change():
    """Atualiza os campos de parâmetros quando o tipo de furo muda."""
    update_hole_parameters()

# Inicializar campos de parâmetros do furo quando a página carrega
update_hole_parameters()

# Handler para exportar a malha em JSON
@when("click", "#export-btn")
def export_mesh():
    global last_mesh_data
    if last_mesh_data is None:
        show_status("Nenhuma malha gerada para exportar!", "error")
        return
    # Converter para JSON
    mesh_json = json.dumps(last_mesh_data, indent=2)
    # Criar um blob e baixar via JS
    blob = Blob.new([mesh_json], {"type": "application/json"})
    url = js.URL.createObjectURL(blob)
    link = document.createElement("a")
    link.href = url
    link.download = "malha.json"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    js.URL.revokeObjectURL(url)

if __name__ == "__main__":
    # Exemplo de uso
    lc = 0.05
    L = 1.0
    H = 0.5
    hole_type = "circle"
    hole_params = {"radius": 0.1}
    
    result = generate_mesh_data(lc, L, H, hole_type, hole_params)
    # print(json.dumps(result, indent=2)) 