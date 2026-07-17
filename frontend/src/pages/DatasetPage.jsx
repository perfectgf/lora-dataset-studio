/**
 * Dataset Maker page — build a face dataset for LoRA character training:
 * generate Klein variations from a reference, import real photos, curate,
 * caption (Qwen3-VL), and export a training-ready ZIP.
 */
import { useDataset } from '../hooks/useDataset';
import DatasetListPanel from '../components/dataset/DatasetListPanel';
import DatasetWorkspace from '../components/dataset/DatasetWorkspace';

export default function DatasetPage() {
  const ds = useDataset();
  return (
    <div className="p-4 max-w-6xl mx-auto">
      {ds.currentId ? (
        <DatasetWorkspace ds={ds} onBack={() => ds.setCurrentId(null)} />
      ) : (
        /* Full page width (max-w-6xl above): the library is a desktop-first
           browsing surface — more columns beat a narrower reading measure.
           The empty-state hero and the creation form re-cap themselves. */
        <DatasetListPanel datasets={ds.datasets} onOpen={ds.open} onCreate={ds.create}
          onDelete={ds.deleteDataset} onRestore={ds.importBackup}
          onExportZip={ds.exportZipFor} onExportBackup={ds.exportBackupFor} />
      )}
    </div>
  );
}
