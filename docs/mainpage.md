## Half-edge mesh

Each vertex knows about its position \f$\mathbf{x}_v\f$ and an outgoing half-edge \f$h^\mathrm{out}_v\f$. Each half-edge knows about its origin vertex \f$v^\mathrm{origin}_h\f$, the next half-edge in the face cycle \f$h^\mathrm{next}_h\f$, the half-edge antiparallel to it \f$h^\mathrm{twin}_h\f$, and the face to its left \f$f^\mathrm{left}_h\f$. We use the convention that \f$f^\mathrm{left}_h<0\f$ means the half-edge is contained in the \f$b=-\left(f^\mathrm{left}_h+1\right)\equiv f^\mathrm{left}_h \left(\mathrm{mod} \left|F\right|\right)\f$ component of \f$\partial M\f$ and is negatively oriented with respect to the surface normal. Each face knows about some half-edge on its boundary \f$h^\mathrm{right}_f\f$, positively oriented with respect to the surface normal. Finally, each boundary component knows about one of its half-edges \f$h^\mathrm{negative}_b\f$ which is negatively oriented with respect to the surface normal. Vertex positions \f$ \lbrace\mathbf{x}_v\rbrace_{v\in V} \f$ as `meshbrane::Samples3d` and each of the maps,
\f[
% h^\mathrm{out}:V \rightarrow H\\
v\mapsto h^\mathrm{out}_v
\\
% v^\mathrm{origin}:H \rightarrow V\\
h\mapsto v^\mathrm{origin}_h
\\
% h^\mathrm{next}:H \rightarrow H\\
h\mapsto h^\mathrm{next}_h
\\
% h^\mathrm{twin}:H \rightarrow H\\
h\mapsto h^\mathrm{twin}_h
\\
% f^\mathrm{left}:H \rightarrow F\\
h\mapsto f^\mathrm{left}_h
\\
% h^\mathrm{right}:F \rightarrow H\\
f\mapsto h^\mathrm{right}_f
\\
% h^\mathrm{right}:B \rightarrow H\\
b\mapsto h^\mathrm{negative}_b
\f]
as `meshbrane::Samplesi`.
