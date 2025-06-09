def linearize(graph):
    """Standalone converter from graph dict to linearized string."""
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    parts = []
    for edge in edges:
        op = edge['op']
        args = []
        for idx in edge['inputs']:
            node = nodes[idx]
            if node.get('type') == 'const':
                args.append(f"const{node['value']}")
            else:
                args.append(f"out{idx}")
        parts.append(f"{op}({','.join(args)})")
    return ' , '.join(parts)