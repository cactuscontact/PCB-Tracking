import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import re


class DXF2IMG(object):
    name = "output_layer.dxf"
    default_img_format = '.png'
    default_img_res = 300
    def convert_dxf2img(self, names, img_format=default_img_format, img_res=default_img_res):
        for name in names:
            doc = ezdxf.readfile(name)
            msp = doc.modelspace()
            doc_layer = msp.query('LINE[layer=="Top"]')
            # Recommended: audit & repair DXF document before rendering
            #auditor = doc_layer.audit()
            # The auditor.errors attribute stores severe errors,
            # which *may* raise exceptions when rendering.
          #  if len(auditor.errors) != 0:
          #      raise Exception("The DXF document is damaged and can't be converted!")
         #   else:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc_layer)
            ctx.set_current_layout(msp)
            #ctx.current_layout.set_colors(bg='#FFFFFF')
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)

            img_name = "output_SingleLayer"  # select the image name that is the same as the dxf file name
            first_param = ''.join(img_name) + img_format  #concatenate list and string
            fig.savefig(first_param, dpi=img_res)


if __name__ == '__main__':
    first = DXF2IMG()
    first.convert_dxf2img(['output_layer.DXF'],img_format='.png')