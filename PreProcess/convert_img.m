function bold_mat = convert_img(img_name)
  load(img_name);
  bold_mat = nii.img;
  name = nii.fileprefix;
  save('subject_name.txt', 'name');
endfunction
