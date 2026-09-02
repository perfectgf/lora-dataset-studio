/**
 * Dataset Maker page — build a face dataset for LoRA character training:
 * generate Klein variations from a reference, import real photos, curate,
 * caption (Qwen3-VL), and export a training-ready ZIP.
 */
import { useDataset } from '../hooks/useDataset';
import DatasetListPanel from '../components/dataset/DatasetListPanel';
import DatasetWorkspace from '../components/dataset/DatasetWorkspace';
import VideoDatasetsPanel from '../components/videobank/VideoDatasetsPanel';

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
        <div className="flex flex-col gap-4">
          <DatasetListPanel datasets={ds.datasets} onOpen={ds.open} onCreate={ds.create}
            onDelete={ds.deleteDataset} onRestore={ds.importBackup}
            onExportZip={ds.exportZipFor} onExportBackup={ds.exportBackupFor}
            backup={{
              start: ds.backupEverything, job: ds.backupJob,
              download: ds.downloadBackup, openFolder: ds.openBackupsFolder,
              dismiss: ds.dismissBackup,
              restoreJob: ds.restoreJob, dismissRestore: ds.dismissRestore,
            }} />
          {/* Video training sets live in the SAME library, below the image ones —
              they are datasets, and a second LIBRARY for them would be a second
              place to remember. That rule was always about the list, and it
              still holds: opening one now goes to its own workspace at
              /video-dataset/<id>, exactly as opening an image dataset swaps this
              page for DatasetWorkspace. One library, two kinds of set, a
              workspace each. */}
          <VideoDatasetsPanel />
        </div>
      )}
    </div>
  );
}
