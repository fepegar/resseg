from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


color_tables_dir = Path(__file__).parent


class Parcellation:
    def __init__(self, parcellation_path):
        self.parcellation_path = Path(parcellation_path)
        self._label_map = None

    @property
    def label_map(self):
        if self._label_map is None:
            label_map_nii = nib.load(self.parcellation_path)
            self._label_map = label_map_nii.get_data().astype(np.uint16)
        return self._label_map

    def get_resected_structures(self, resection_seg_path, ignore=None):
        if ignore is None:
            ignore = []
        resected_dict = self.get_resected_labels_and_counts(resection_seg_path)
        structures = []
        for resected_number, num_voxels_resected in resected_dict.items():
            structure_name = self.color_table.get_structure_from_label_number(
                resected_number)
            ignore_this = False
            for substring_to_ignore in ignore:
                if substring_to_ignore in structure_name:
                    ignore_this = True
                    break
            if ignore_this:
                continue
            num_voxels_parcellation = np.count_nonzero(
                self.label_map == resected_number)
            ratio = num_voxels_resected / num_voxels_parcellation
            structures.append((
                structure_name,
                num_voxels_resected,
                ratio,
            ))
        return list(zip(*structures))

    def get_resected_labels_and_counts(self, resection_seg_path):
        mask_nii = nib.load(resection_seg_path)
        mask = mask_nii.get_data() > 0
        masked_values = self.label_map[mask]

        unique, counts = np.unique(masked_values, return_counts=True)
        resected_dict = dict(zip(unique, counts))
        return resected_dict

    def print_percentage_of_resected_structures(self,
                                                resection_seg_path,
                                                hide_zeros=True):
        structures, voxels, ratios = self.get_resected_structures(
            resection_seg_path)
        sort_by_ratio = np.argsort(ratios)
        print('Percentage of each resected structure:')
        for idx in reversed(sort_by_ratio):
            ratio = ratios[idx]
            structure = structures[idx]
            percentage = int(ratio * 100)
            if percentage == 0 and hide_zeros:
                continue
            structure_pretty = structure.replace('-', ' ')
            print(f'{percentage:3}% of {structure_pretty}')
        print()

        sort_by_voxels = np.argsort(voxels)
        total_voxels = sum(voxels)
        print('The resection volume is composed of:')
        for idx in reversed(sort_by_voxels):
            ratio = voxels[idx] / total_voxels
            structure = structures[idx]
            percentage = int(ratio * 100)
            if percentage == 0 and hide_zeros:
                continue
            structure_pretty = structure.replace('-', ' ')
            print(f'{percentage:3}% is {structure_pretty}')

    def plot_pie(
            self,
            resection_seg_path,
            title=None,
            show=True,
            pct_threshold=2,
            output_path=None,
            ignore=None,
            ):
        names, voxels, _ = self.get_resected_structures(
            resection_seg_path, ignore=ignore)
        colors = [
            self.color_table.get_color_from_structure_name(name)
            for name in names
        ]
        fig, ax = plt.subplots()
        sort_by_voxels = np.argsort(voxels)[::-1]  # descending order
        voxels = np.array(voxels)[sort_by_voxels]
        percentages = (voxels / voxels.sum()) * 100
        names = np.array(names)[sort_by_voxels]
        colors = np.array(colors)[sort_by_voxels]

        # Hide some values
        def my_autopct(pct):
            return f'{int(pct)}%' if pct > pct_threshold else ''

        labels = names[:]
        for i, pct in enumerate(percentages):
            if pct <= pct_threshold:
                labels[i] = ''

        ax.pie(
            percentages,
            labels=labels,
            colors=colors,
            shadow=False,
            autopct=my_autopct,
            pctdistance=0.7,
        )
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=400)
        if show:
            plt.show()
        return fig

    def plot_bars(
            self,
            resection_seg_path,
            title=None,
            show=True,
            output_path=None,
            ignore=None,
            ):
        names, _, ratios = self.get_resected_structures(
            resection_seg_path, ignore=ignore)

        colors = [
            self.color_table.get_color_from_structure_name(name)
            for name in names
        ]
        fig, ax = plt.subplots()
        sort_by_ratios = np.argsort(ratios)
        ratios = np.array(ratios)[sort_by_ratios]
        percentages = ratios * 100
        names = np.array(names)[sort_by_ratios]
        colors = np.array(colors)[sort_by_ratios]

        y_pos = np.arange(len(names))
        ax.barh(
            y_pos,
            percentages,
            align='center',
            color=colors,
            tick_label=names,
        )
        ax.set_axisbelow(True)  # https://stackoverflow.com/a/39039520
        ax.grid()
        ax.set_xlim((0, 105))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        if title is not None:
            ax.set_title(title)

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=400)
        if show:
            plt.show()
        return fig

    def is_valid_number(self, number):
        return self.color_table.is_valid_number(number)


class GIFParcellation(Parcellation):
    def __init__(self, parcellation_path):
        Parcellation.__init__(self, parcellation_path)
        self.color_table = GIFColorTable()


class FreeSurferParcellation(Parcellation):
    def __init__(self, parcellation_path):
        Parcellation.__init__(self, parcellation_path)
        self.color_table = FreeSurferColorTable()


class ColorTable:
    def __init__(self):
        self.fieldnames = (
            'structure',
            'red',
            'green',
            'blue',
            'alpha',
        )

    def get_value_from_label_number(self, label_number, key):
        try:
            value = self._data_frame.loc[label_number][key]
        except KeyError:
            value = f'[Unkown label: {label_number}]'
        return value

    def get_row_from_structure_name(self, name):
        mask = self._data_frame['structure'] == name
        row = self._data_frame.loc[mask]
        return row

    def get_value_from_structure_name(self, name, key):
        row = self.get_row_from_structure_name(name)
        value = row[key]
        return value

    def get_structure_from_label_number(self, label_number):
        return self.get_value_from_label_number(label_number, 'structure')

    def get_color_from_structure_name(self, name):
        row = self.get_row_from_structure_name(name)
        if row.empty:
            color = 0, 0, 0
        else:
            color = [row[c].values for c in ('red', 'green', 'blue')]
            color = np.hstack(color)
            color = np.array(color) / 255
        return color

    def is_valid_number(self, number):
        return number in self._data_frame.index


class GIFColorTable(ColorTable):
    def __init__(self):
        ColorTable.__init__(self)
        self.color_table_path = color_tables_dir / 'BrainAnatomyLabelsV3_0.txt'
        self._data_frame = self.read_color_table()

    def read_color_table(self):
        df = pd.read_csv(
            self.color_table_path,
            index_col=0,
            names=self.fieldnames,
            sep=r'\s+',  # there is a double space in the file
        )
        return df


class FreeSurferColorTable(ColorTable):
    def __init__(self):
        ColorTable.__init__(self)
        self.color_table_path = color_tables_dir / 'FreeSurferLabels.ctbl'
        self._data_frame = self.read_color_table()

    def read_color_table(self):
        df = pd.read_csv(
            self.color_table_path,
            index_col=0,
            names=self.fieldnames,
            sep=r'\s+',
            skiprows=2,
        )
        return df
