import DatasetGridItem from './DatasetGridItem';

export default function DatasetGrid({ images, datasetId, onStatus, onCaption, onCrop, onDelete,
                                      onRegenerate, onView, nonces }) {
  if (!images || !images.length) {
    return <p className="text-content-subtle text-xs">No images — generate variations or import photos.</p>;
  }
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
      {images.map((img) => (
        <DatasetGridItem key={img.id} img={img} datasetId={datasetId} onStatus={onStatus} onCaption={onCaption}
          onCrop={onCrop} onDelete={onDelete} onRegenerate={onRegenerate} onView={onView}
          nonce={(nonces && nonces[img.id]) || 0} />
      ))}
    </div>
  );
}
