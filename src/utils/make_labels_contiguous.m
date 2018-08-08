function labels = make_labels_contiguous(labels, classes)

nb_classes = length(classes);
classes_idx = 1:nb_classes;
for i=1:nb_classes
    labels(labels==classes(i))=classes_idx(i);
end

end
