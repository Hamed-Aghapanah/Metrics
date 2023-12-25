import MAL.losses.segment as l
import MAL.phantom as ph


gt = ph.Rectangle(n_classes=1)()

pr = ph.Zeros(n_classes=1)()

print(
    l.dice()(gt, pr)
)