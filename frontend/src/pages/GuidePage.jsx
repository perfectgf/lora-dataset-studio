import Markdown from '../components/common/Markdown';
// Source unique : docs/DATASET_GUIDE.md (aussi lisible sur GitHub). Vite inline
// le fichier en string au build (?raw) → la page vit dans le bundle, aucun
// fetch ni fichier à embarquer dans le bundle portable.
import guideMd from '../../../docs/DATASET_GUIDE.md?raw';

export default function GuidePage() {
  return (
    <div className="max-w-3xl mx-auto pb-10">
      <Markdown source={guideMd} />
    </div>
  );
}
