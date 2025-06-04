import plotly.graph_objects as go, plotly.io as pio, pathlib
pio.renderers.default = "browser"          # make sure browser renderer exists
pio.kaleido.scope.default_format = "png"   # force Kaleido

fig = go.Figure(data=go.Bar(y=[2,3,1]))
out = pathlib.Path("kaleido_ping.png")
fig.write_image(out, width=400, height=300, engine="kaleido")
print("wrote", out.resolve())
