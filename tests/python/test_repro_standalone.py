# Add the project root to the path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nvfuser import FusionDefinition


class MyFusion(FusionDefinition):
    def definition(self):
        return;

fd = MyFusion()
