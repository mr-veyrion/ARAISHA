from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple
from datetime import datetime

from .graph_store import GraphStore, Relation


def export_pyvis(
    graph: GraphStore,
    nodes: Optional[Iterable[str]] = None,
    edges: Optional[Iterable[Relation]] = None,
    out_dir: str = os.path.join(os.getcwd(), "local_data", "viz"),
    filename_prefix: str = "graph_view",
    faiss_memories: Optional[dict] = None,  # memory_id -> memory text mapping
) -> str:
    """Export graph to interactive HTML visualization.
    
    Args:
        graph: GraphStore instance
        nodes: Optional list of node names to include
        edges: Optional list of Relation edges to include
        out_dir: Output directory path
        filename_prefix: Prefix for output file name
        faiss_memories: Optional dict mapping memory_id to FAISS memory text (shown on node click)
    """
    try:
        from pyvis.network import Network
    except Exception:
        raise RuntimeError("pyvis is not installed. Install with: pip install pyvis")

    os.makedirs(out_dir, exist_ok=True)
    
    # Create network with modern styling
    net = Network(
        height="700px", 
        width="100%", 
        directed=True, 
        notebook=False,
        bgcolor="#1a1a1a",  # Dark background
        font_color="#ffffff"  # White text
    )
    
    # Configure physics for better layout
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "layout": {
        "improvedLayout": true
      },
      "interaction": {
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false
      }
    }
    """)

    nodes = list(nodes or [])
    seen = set()

    # Modern color palette (dim, beautiful colors)
    node_colors = [
        "#64b5f6",  # Light blue
        "#81c784",  # Light green  
        "#ffb74d",  # Light orange
        "#e57373",  # Light red
        "#ba68c8",  # Light purple
        "#4db6ac",  # Teal
        "#aed581",  # Light lime
        "#ff8a65",  # Deep orange
        "#9575cd",  # Deep purple
        "#4fc3f7",  # Light cyan
        "#dce775",  # Lime
        "#f06292",  # Pink
        "#90a4ae",  # Blue grey
        "#a1887f",  # Brown
        "#ffd54f"   # Amber
    ]
    
    relationship_colors = [
        "#42a5f5",  # Blue for work/professional
        "#66bb6a",  # Green for family/personal
        "#ff7043",  # Orange for actions
        "#ab47bc",  # Purple for emotions
        "#26c6da",  # Cyan for locations
        "#78909c",  # Grey for misc
    ]

    # If no edges provided, fetch neighbors for provided nodes (CASE-INSENSITIVE)
    if not edges and nodes:
        rels: List[Relation] = []
        seen_edges_for_nodes = set()
        
        for requested_node in nodes:
            found_matches = False
            
            # Get all edges from graph
            all_edges = graph.query_edges()
            
            # Find case-insensitive matches
            for edge in all_edges:
                if (edge.source.upper() == requested_node.upper() or 
                    edge.destination.upper() == requested_node.upper()):
                    edge_key = (edge.source.upper(), edge.relationship.upper(), edge.destination.upper())
                    if edge_key not in seen_edges_for_nodes:
                        rels.append(edge)
                        seen_edges_for_nodes.add(edge_key)
                        found_matches = True
            
            # If no matches found, try getting neighbors by exact name (fallback)
            if not found_matches:
                try:
                    node_rels = graph.neighbors(requested_node, direction="both", relationship=None, limit=200)
                    for r in node_rels:
                        edge_key = (r.source.upper(), r.relationship.upper(), r.destination.upper())
                        if edge_key not in seen_edges_for_nodes:
                            rels.append(r)
                            seen_edges_for_nodes.add(edge_key)
                except Exception:
                    pass
        
        edges = rels

    # Deduplicate identical edges (source, relationship, destination)
    edge_counts: dict[Tuple[str, str, str], int] = {}
    edge_examples: dict[Tuple[str, str, str], Relation] = {}
    node_memory_ids: dict[str, set] = {}  # node -> set of memory_ids
    
    if edges:
        for r in edges:
            key = (r.source.upper(), r.relationship.upper(), r.destination.upper())
            edge_counts[key] = edge_counts.get(key, 0) + 1
            if key not in edge_examples:
                edge_examples[key] = r
            # Collect memory_ids for nodes
            if r.memory_id:
                node_memory_ids.setdefault(r.source, set()).add(r.memory_id)
                node_memory_ids.setdefault(r.destination, set()).add(r.memory_id)

    # Build node tooltips from FAISS memories
    node_tooltips: dict[str, str] = {}
    if faiss_memories:
        for node, mem_ids in node_memory_ids.items():
            memories = []
            for mid in mem_ids:
                if mid in faiss_memories:
                    memories.append(faiss_memories[mid])
            if memories:
                # Create tooltip with memories (limit to avoid huge tooltips)
                tooltip_text = "\\n---\\n".join(memories[:5])
                if len(memories) > 5:
                    tooltip_text += f"\\n... and {len(memories) - 5} more"
                node_tooltips[node] = tooltip_text

    # Add nodes with modern styling
    node_color_map = {}
    color_index = 0
    
    if edge_counts:
        for (s_upper, rel_upper, d_upper), count in edge_counts.items():
            # Get original case from example
            example = edge_examples[(s_upper, rel_upper, d_upper)]
            s, rel, d = example.source, example.relationship, example.destination
            
            # Assign colors to nodes
            if s not in node_color_map:
                node_color_map[s] = node_colors[color_index % len(node_colors)]
                color_index += 1
            if d not in node_color_map:
                node_color_map[d] = node_colors[color_index % len(node_colors)]
                color_index += 1
            
            # Add nodes with beautiful styling (title shows FAISS memory on hover)
            if s not in seen:
                node_title = node_tooltips.get(s)
                net.add_node(
                    s, 
                    label=s,
                    title=node_title,  # FAISS memory shown on hover
                    color=node_color_map[s],
                    size=25,  # Larger nodes
                    font={"size": 14, "color": "#ffffff", "strokeWidth": 2, "strokeColor": "#000000"},
                    borderWidth=2,
                    borderWidthSelected=3,
                    shadow={"enabled": True, "color": "rgba(0,0,0,0.3)", "size": 10, "x": 3, "y": 3}
                )
                seen.add(s)
                
            if d not in seen:
                node_title = node_tooltips.get(d)
                net.add_node(
                    d, 
                    label=d,
                    title=node_title,  # FAISS memory shown on hover
                    color=node_color_map[d], 
                    size=25,  # Larger nodes
                    font={"size": 14, "color": "#ffffff", "strokeWidth": 2, "strokeColor": "#000000"},
                    borderWidth=2,
                    borderWidthSelected=3,
                    shadow={"enabled": True, "color": "rgba(0,0,0,0.3)", "size": 10, "x": 3, "y": 3}
                )
                seen.add(d)
            
            # Style edges with modern colors and reduced thickness
            label = rel if count == 1 else f"{rel} ×{count}"
            
            # Assign relationship colors based on type
            rel_color = "#64b5f6"  # Default blue
            rel_lower = rel.lower()
            if any(word in rel_lower for word in ["work", "job", "employ", "company"]):
                rel_color = "#42a5f5"  # Blue for work
            elif any(word in rel_lower for word in ["love", "like", "marry", "family", "friend"]):
                rel_color = "#66bb6a"  # Green for relationships
            elif any(word in rel_lower for word in ["live", "location", "city", "place"]):
                rel_color = "#26c6da"  # Cyan for locations
            elif any(word in rel_lower for word in ["feel", "emotion", "happy", "sad"]):
                rel_color = "#ab47bc"  # Purple for emotions
            else:
                rel_color = relationship_colors[hash(rel) % len(relationship_colors)]
            
            # Add edge with improved styling
            width = min(max(1.5, count * 0.8), 4)  # Thinner edges, max width 4
            net.add_edge(
                s, d,
                label=label,
                width=width,
                color={"color": rel_color, "opacity": 0.7},
                font={"size": 12, "color": "#ffffff", "strokeWidth": 1, "strokeColor": "#000000"},
                arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
                length=150,  # Longer edges
                smooth={"enabled": True, "type": "curvedCW", "roundness": 0.2}
            )

    # Add any extra nodes not in edges
    for n in nodes:
        if n not in seen:
            if n not in node_color_map:
                node_color_map[n] = node_colors[color_index % len(node_colors)]
                color_index += 1
            
            net.add_node(
                n, 
                label=n,
                color=node_color_map[n],
                size=25,
                font={"size": 14, "color": "#ffffff", "strokeWidth": 2, "strokeColor": "#000000"},
                borderWidth=2,
                shadow={"enabled": True, "color": "rgba(0,0,0,0.3)", "size": 10, "x": 3, "y": 3}
            )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.html")
    
    # Write HTML with custom styling
    try:
        net.write_html(out_path, notebook=False)
        
        # Add custom CSS for better appearance
        with open(out_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Insert custom CSS
        css_style = """
        <style>
        body { 
            margin: 0; 
            padding: 10px; 
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        #mynetworkid {
            border: 2px solid #404040;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .vis-tooltip {
            background: rgba(50,50,50,0.9) !important;
            color: white !important;
            border: 1px solid #666 !important;
            border-radius: 5px !important;
            font-size: 12px !important;
        }
        </style>
        """
        
        content = content.replace('<head>', f'<head>{css_style}')
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception:
        # fallback to show if write_html unavailable
        net.show(out_path)
        
    return out_path


