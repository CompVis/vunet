from batches_pg2_vis import get_batches, plot_batch, postprocess

    elif opt.mode == "transfer":
        if not opt.checkpoint:
            opt.checkpoint = "log/2017-10-04T16:01:46/checkpoints/model.ckpt-100000"
        batch_size = opt.batch_size
        img_shape = 2*[opt.spatial_size] + [3]
        data_shape = [batch_size] + img_shape
        valid_batches = get_batches(data_shape, opt.data_index,
                mask = opt.mask, train = False)
        model = Model(opt, out_dir, logger)
        model.restore_graph(opt.checkpoint)

        ids = ["00038", "00281", "01166", "x", "06909", "y", "07586", "07607", "z", "09874"]
        for step in trange(10):
            X_batch, C_batch = next(valid_batches)
            bs = X_batch.shape[0]
            imgs = list()
            imgs.append(np.zeros_like(X_batch[0,...]))
            for r in range(bs):
                imgs.append(C_batch[r,...])
            for i in range(bs):
                x_infer = X_batch[i,...]
                c_infer = C_batch[i,...]
                imgs.append(x_infer)

                x_infer_batch = x_infer[None,...].repeat(bs, axis = 0)
                c_infer_batch = c_infer[None,...].repeat(bs, axis = 0)
                c_generate_batch = C_batch
                results = model.transfer(x_infer_batch, c_infer_batch, c_generate_batch)
                for j in range(bs):
                    imgs.append(results[j,...])
            imgs = np.stack(imgs, axis = 0)
            plot_batch(imgs, os.path.join(
                out_dir,
                "transfer_{}.png".format(ids[step])))
