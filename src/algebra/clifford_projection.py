from algebra.cliffordalgebra import CliffordAlgebra
from engineer.metrics.metrics import Loss, MetricCollection
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear
from models.modules.normalization import NormalizationLayer


### equivariant projection layers
class CliffordProjection(nn.Module):
    def __init__(self, input_dim=3, feature_dim=512, output_dim=512):
        super(CliffordProjection, self).__init__()

        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))

        in_features = input_dim + feature_dim
        hidden_features = output_dim

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        self.geometric_product = SteerableGeometricProductLayer(
            algebra=self.algebra, features=hidden_features
        )
        self.normalization = NormalizationLayer(
            algebra=self.algebra, features=hidden_features
        )

    def forward(self, x, f):
        q = x
        k = x
        v = x

        y = x

        ### geometric linear layer
        y = self.embedding(y)

        ### geometric product
        y = self.geometric_product(y)

        ### normilization
        y = self.normalization(y)

        """ Finally, the
            output is projected back to scalar and vector features by
            grade-one and grade-two projections respectively"""

        z_inv, z_eqv = y, y

        return q, k, v
