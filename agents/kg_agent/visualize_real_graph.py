import networkx as nx
import matplotlib.pyplot as plt
from kg_real_traversal import build_graph, traverse

G = build_graph()
output = traverse(G, "myocardial infarction", ["Sildenafil"])

# For readability, show only a handful of the 16 indirect treatment paths:
# the case-study drug (Nitroglycerin, highlighted) + 3 other well-known
# MI treatments for context (dimmed), rather than all 16.
context_treatments = ["Nitroglycerin", "Acetylsalicylic acid", "Metoprolol", "Atenolol"]

fig, ax = plt.subplots(figsize=(13, 10))

nodes = {"myocardial infarction": "disease", "Sildenafil": "drug"}
edges_to_draw = []  # (u, v, kind)  kind: 'direct' | 'indirect_focus' | 'indirect_context'

# direct contraindication edge
edges_to_draw.append(("Sildenafil", "myocardial infarction", "direct"))

for t in context_treatments:
    if G.has_node(t):
        nodes[t] = "drug"
        # treatment -> MI edge (indication/off-label)
        if G.has_edge(t, "myocardial infarction"):
            edges_to_draw.append((t, "myocardial infarction", "context"))
        # treatment -> Sildenafil drug_drug edge
        if G.has_edge(t, "Sildenafil"):
            kind = "indirect_focus" if t == "Nitroglycerin" else "indirect_context"
            edges_to_draw.append((t, "Sildenafil", kind))

pos = {
    "myocardial infarction": (0, 0),
    "Sildenafil": (6, 0),
    "Nitroglycerin": (3, 2.6),
    "Acetylsalicylic acid": (3, 1.1),
    "Metoprolol": (3, -1.1),
    "Atenolol": (3, -2.6),
}

type_colors = {"drug": "#97C459", "disease": "#F0997B"}

for n, ntype in nodes.items():
    x, y = pos[n]
    ax.scatter([x], [y], s=2600, color=type_colors[ntype],
               edgecolors="#D1332C" if n in ("Sildenafil", "myocardial infarction", "Nitroglycerin") else "#444441",
               linewidths=3 if n in ("Sildenafil", "myocardial infarction", "Nitroglycerin") else 1, zorder=3)
    ax.annotate(n, (x, y), ha="center", va="center", fontsize=9, fontweight="bold", zorder=4,
                wrap=True)

style_map = {
    "direct": dict(color="#D1332C", lw=3.2, ls="-"),
    "indirect_focus": dict(color="#D1332C", lw=2.6, ls="--"),
    "context": dict(color="#c7c7c7", lw=1.0, ls="-"),
    "indirect_context": dict(color="#c7c7c7", lw=1.0, ls="--"),
}

for u, v, kind in edges_to_draw:
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    st = style_map[kind]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=st["lw"],
                                 linestyle=st["ls"], shrinkA=45, shrinkB=45,
                                 connectionstyle="arc3,rad=0.08"), zorder=2)

ax.text(0, 0.55, "contraindication", fontsize=8, color="#D1332C", ha="center")
ax.text(4.7, 1.7, "synergistic\ninteraction", fontsize=8, color="#D1332C", ha="center")
ax.text(1.3, 1.9, "off-label use", fontsize=7.5, color="#888880", ha="center")

legend_elems = [
    plt.Line2D([0], [0], marker='o', color='w', label='Drug', markerfacecolor='#97C459', markersize=14),
    plt.Line2D([0], [0], marker='o', color='w', label='Disease', markerfacecolor='#F0997B', markersize=14),
    plt.Line2D([0], [0], color='#D1332C', lw=3, label='Real PrimeKG danger paths (this patient)'),
    plt.Line2D([0], [0], color='#c7c7c7', lw=1.5, label='Other MI treatments (context only)'),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0.0, 1.05))

ax.set_title("Real PrimeKG data \u2014 candidate diagnosis: Myocardial Infarction\nPatient medication: Sildenafil",
              fontsize=13, fontweight="bold")
ax.set_xlim(-2.5, 8.5)
ax.set_ylim(-3.5, 3.5)
ax.axis("off")
plt.tight_layout()
plt.savefig("kg_real_visual.png", dpi=180, bbox_inches="tight")




print("saved")
print()
print("Direct flags:", len(output["direct_flags"]))
print("Indirect flags (all MI treatments w/ generic drug_drug edge to Sildenafil):", len(output["indirect_flags"]))
